import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

DB_PATH = "companies.db"
_openai: Optional[AsyncOpenAI] = None


def get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured in .env")
        _openai = AsyncOpenAI(api_key=api_key)
    return _openai


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pipeline (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id  INTEGER NOT NULL REFERENCES companies(id),
                status      TEXT    NOT NULL DEFAULT 'prospect',
                notes       TEXT,
                date_added  TEXT    NOT NULL DEFAULT (date('now'))
            )
        """)
        try:
            await db.execute("ALTER TABLE companies ADD COLUMN ai_summary TEXT")
        except Exception:
            pass  # column already exists
        await db.commit()
    yield


app = FastAPI(title="Search Fund Acquisition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------

def row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def _calc_years(date_of_inc) -> Optional[int]:
    if not date_of_inc:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y"):
        try:
            inc = datetime.strptime(str(date_of_inc).strip(), fmt)
            return int((datetime.now() - inc).days / 365.25)
        except ValueError:
            continue
    return None


# ---------- schemas ----------

class PipelineIn(BaseModel):
    company_id: int
    status: Optional[str] = "prospect"
    notes: Optional[str] = None
    date_added: Optional[date] = None


class PipelineOut(PipelineIn):
    id: int
    company_name: Optional[str] = None
    acquisition_score: Optional[int] = None
    industry_sector: Optional[str] = None
    state: Optional[str] = None


class NLSearchQuery(BaseModel):
    query: str


# ---------- routes ----------

@app.get("/companies")
async def list_companies(
    industry: Optional[str] = Query(None, description="Filter by industry_sector (partial match)"),
    state: Optional[str] = Query(None, description="Filter by state (exact, e.g. TX)"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum acquisition_score"),
    max_employees: Optional[int] = Query(None, ge=0, description="Maximum employee_count"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    conditions = ["business_status = 'active'"]
    params: list = []

    if industry:
        conditions.append("industry_sector LIKE ?")
        params.append(f"%{industry}%")
    if state:
        conditions.append("state = ?")
        params.append(state.upper())
    if min_score is not None:
        conditions.append("acquisition_score >= ?")
        params.append(min_score)
    if max_employees is not None:
        conditions.append(
            "(employee_count IS NULL OR CAST(REPLACE(employee_count, ',', '') AS INTEGER) <= ?)"
        )
        params.append(max_employees)

    where = " AND ".join(conditions)
    offset = (page - 1) * page_size

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(f"SELECT COUNT(*) FROM companies WHERE {where}", params) as cur:
            total = (await cur.fetchone())[0]

        async with db.execute(
            f"""
            SELECT id, company_name, industry_sector, naics_code, sic_code,
                   city, state, state_of_inc, zip_code, employee_count,
                   award_amount, business_type, source, acquisition_score,
                   cluster_label
            FROM companies
            WHERE {where}
            ORDER BY acquisition_score DESC NULLS LAST, id
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, -(-total // page_size)),
        "results": rows,
    }


@app.get("/companies/{company_id}/summary")
async def get_company_summary(company_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM companies WHERE id = ?", (company_id,)) as cur:
            row = await cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Company not found")

    company = dict(row)

    if company.get("ai_summary"):
        return {"summary": company["ai_summary"], "cached": True}

    years = _calc_years(company.get("date_of_inc"))
    years_str = f"{years}" if years is not None else "N/A"
    has_gov = "Yes" if (company.get("award_amount") or 0) > 0 else "No"
    location = ", ".join(filter(None, [company.get("city"), company.get("state")])) or "Unknown"

    user_prompt = (
        "Write a 3-sentence acquisition assessment for a search fund operator "
        "considering buying this company:\n"
        f"Company: {company.get('company_name', 'Unknown')}\n"
        f"Industry: {company.get('industry_sector', 'Unknown')}\n"
        f"Location: {location}\n"
        f"Years in business: {years_str} years\n"
        f"Acquisition score: {company.get('acquisition_score', 0)}/100\n"
        f"Government contracts: {has_gov}\n"
        f"Cluster: {company.get('cluster_label', 'Unknown')}\n\n"
        "Mention the strongest signal, the biggest risk, and one specific question "
        "the buyer should investigate."
    )

    client = get_openai()
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a senior analyst at a search fund. Write concise, professional acquisition assessments.",
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=220,
        temperature=0.7,
    )

    summary = response.choices[0].message.content.strip()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE companies SET ai_summary = ? WHERE id = ?", (summary, company_id)
        )
        await db.commit()

    return {"summary": summary, "cached": False}


@app.get("/companies/{company_id}")
async def get_company(company_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM companies WHERE id = ?", (company_id,)
        ) as cur:
            row = await cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return dict(row)


@app.post("/pipeline", status_code=201)
async def add_to_pipeline(payload: PipelineIn):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT id FROM companies WHERE id = ?", (payload.company_id,)
        ) as cur:
            if await cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Company not found")

        date_str = str(payload.date_added) if payload.date_added else None
        async with db.execute(
            """
            INSERT INTO pipeline (company_id, status, notes, date_added)
            VALUES (?, ?, ?, COALESCE(?, date('now')))
            """,
            (payload.company_id, payload.status, payload.notes, date_str),
        ) as cur:
            new_id = cur.lastrowid

        await db.commit()

        async with db.execute(
            "SELECT * FROM pipeline WHERE id = ?", (new_id,)
        ) as cur:
            row = dict(await cur.fetchone())

    return row


@app.get("/pipeline")
async def get_pipeline():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT p.id, p.company_id, p.status, p.notes, p.date_added,
                   c.company_name, c.acquisition_score, c.industry_sector,
                   c.city, c.state, c.award_amount
            FROM pipeline p
            JOIN companies c ON c.id = p.company_id
            ORDER BY p.date_added DESC, p.id DESC
            """
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    return {"total": len(rows), "results": rows}


@app.post("/search/natural")
async def natural_language_search(payload: NLSearchQuery):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    client = get_openai()
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a search fund acquisition filter parser. Return only valid JSON, no explanation.",
            },
            {
                "role": "user",
                "content": (
                    "Convert this search fund acquisition search query into filter parameters. "
                    "Return only valid JSON with these optional fields: "
                    "industry (string), state (string, 2-letter code), "
                    "min_score (integer 0-100), max_employees (integer), min_years (integer).\n"
                    f"Query: {payload.query}"
                ),
            },
        ],
        max_tokens=150,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        filters = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        filters = json.loads(match.group()) if match else {}

    conditions = ["business_status = 'active'"]
    params: list = []

    if filters.get("industry"):
        conditions.append("industry_sector LIKE ?")
        params.append(f"%{filters['industry']}%")
    if filters.get("state"):
        conditions.append("state = ?")
        params.append(str(filters["state"]).upper()[:2])
    if filters.get("min_score") is not None:
        try:
            conditions.append("acquisition_score >= ?")
            params.append(int(filters["min_score"]))
        except (ValueError, TypeError):
            pass
    if filters.get("max_employees") is not None:
        try:
            conditions.append(
                "(employee_count IS NULL OR CAST(REPLACE(employee_count, ',', '') AS INTEGER) <= ?)"
            )
            params.append(int(filters["max_employees"]))
        except (ValueError, TypeError):
            pass
    # min_years omitted — date_of_inc is not populated in this dataset

    where = " AND ".join(conditions)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"""
            SELECT id, company_name, industry_sector, naics_code, sic_code,
                   city, state, state_of_inc, zip_code, employee_count,
                   award_amount, business_type, source, acquisition_score,
                   cluster_label
            FROM companies
            WHERE {where}
            ORDER BY acquisition_score DESC NULLS LAST, id
            LIMIT 200
            """,
            params,
        ) as cur:
            results = [dict(r) for r in await cur.fetchall()]

    return {
        "query": payload.query,
        "filters_applied": filters,
        "total": len(results),
        "results": results,
    }
