import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

import anthropic
import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

DB_PATH = "companies.db"
_anthropic: Optional[anthropic.Anthropic] = None

# ---------- state normalization ----------

_STATE_NAMES: dict[str, str] = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
    "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
    "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS",
    "kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA",
    "michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT",
    "nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND",
    "ohio":"OH","oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI",
    "south carolina":"SC","south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT",
    "vermont":"VT","virginia":"VA","washington":"WA","west virginia":"WV",
    "wisconsin":"WI","wyoming":"WY","washington d.c.":"DC","district of columbia":"DC",
}

_REGIONS: dict[str, list[str]] = {
    "southeast":  ["FL","GA","AL","MS","TN","SC","NC","VA"],
    "midwest":    ["OH","IN","IL","MI","WI","MN","IA","MO"],
    "southwest":  ["TX","AZ","NM","CO","NV"],
    "northeast":  ["NY","PA","NJ","MA","CT","ME","VT","NH","RI"],
    "west":       ["CA","OR","WA","ID","MT","WY","UT","NV","AZ"],
    "south":      ["FL","GA","AL","MS","TN","SC","NC","VA","TX","LA","AR","KY","WV"],
    "plains":     ["KS","NE","SD","ND","OK","IA","MO"],
    "mid-atlantic":["NY","NJ","PA","DE","MD","VA","DC"],
}


def _normalize_states(raw) -> list[str]:
    """Return a deduplicated list of 2-letter state abbreviations from any state value Claude returns."""
    if not raw:
        return []
    if isinstance(raw, list):
        result = []
        for item in raw:
            result.extend(_normalize_states(item))
        return list(dict.fromkeys(result))
    s = str(raw).strip().lower()
    if s in _REGIONS:
        return _REGIONS[s]
    if s in _STATE_NAMES:
        return [_STATE_NAMES[s]]
    abbrev = s.upper()
    if len(abbrev) == 2 and abbrev.isalpha():
        return [abbrev]
    return []


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences Claude sometimes wraps JSON in."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_anthropic() -> anthropic.Anthropic:
    global _anthropic
    if _anthropic is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured in .env")
        _anthropic = anthropic.Anthropic(api_key=api_key)
    return _anthropic


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
        for col, defn in [
            ("stage",            "TEXT DEFAULT 'Identified'"),
            ("follow_up_date",   "TEXT"),
            ("owner_name",       "TEXT"),
            ("owner_phone",      "TEXT"),
            ("owner_email",      "TEXT"),
            ("owner_linkedin",   "TEXT"),
            ("outreach_message", "TEXT"),
        ]:
            try:
                await db.execute(f"ALTER TABLE pipeline ADD COLUMN {col} {defn}")
            except Exception:
                pass
        try:
            await db.execute("ALTER TABLE companies ADD COLUMN ai_summary TEXT")
        except Exception:
            pass
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
    stage: Optional[str] = "Identified"
    follow_up_date: Optional[str] = None
    owner_name: Optional[str] = None
    owner_phone: Optional[str] = None
    owner_email: Optional[str] = None
    owner_linkedin: Optional[str] = None
    outreach_message: Optional[str] = None


class PipelineUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    stage: Optional[str] = None
    follow_up_date: Optional[str] = None
    owner_name: Optional[str] = None
    owner_phone: Optional[str] = None
    owner_email: Optional[str] = None
    owner_linkedin: Optional[str] = None
    outreach_message: Optional[str] = None


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

    client = get_anthropic()
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="You are a senior analyst at a search fund. Write concise, professional acquisition assessments.",
        messages=[{"role": "user", "content": user_prompt}],
    )

    summary = response.content[0].text.strip()

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
            INSERT INTO pipeline (company_id, status, notes, date_added,
                                  stage, follow_up_date, owner_name, owner_phone,
                                  owner_email, owner_linkedin, outreach_message)
            VALUES (?, ?, ?, COALESCE(?, date('now')), ?, ?, ?, ?, ?, ?, ?)
            """,
            (payload.company_id, payload.status, payload.notes, date_str,
             payload.stage or 'Identified', payload.follow_up_date,
             payload.owner_name, payload.owner_phone, payload.owner_email,
             payload.owner_linkedin, payload.outreach_message),
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
                   p.stage, p.follow_up_date, p.owner_name, p.owner_phone,
                   p.owner_email, p.owner_linkedin, p.outreach_message,
                   c.company_name, c.acquisition_score, c.industry_sector,
                   c.city, c.state, c.award_amount, c.cluster_label,
                   c.date_of_inc, c.naics_code
            FROM pipeline p
            JOIN companies c ON c.id = p.company_id
            ORDER BY p.date_added DESC, p.id DESC
            """
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    return {"total": len(rows), "results": rows}


@app.patch("/pipeline/{company_id}")
async def update_pipeline(company_id: int, payload: PipelineUpdate):
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [company_id]

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"UPDATE pipeline SET {set_clause} WHERE company_id = ?", values
        ) as cur:
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Pipeline entry not found")
        await db.commit()

        async with db.execute(
            """
            SELECT p.id, p.company_id, p.status, p.notes, p.date_added,
                   p.stage, p.follow_up_date, p.owner_name, p.owner_phone,
                   p.owner_email, p.owner_linkedin, p.outreach_message,
                   c.company_name, c.acquisition_score, c.industry_sector,
                   c.city, c.state, c.award_amount
            FROM pipeline p JOIN companies c ON c.id = p.company_id
            WHERE p.company_id = ?
            """, (company_id,)
        ) as cur:
            row = await cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Pipeline entry not found")
    return dict(row)


@app.get("/pipeline/{company_id}/outreach")
async def generate_outreach(company_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT p.outreach_message, c.company_name, c.industry_sector,
                   c.city, c.state, c.date_of_inc, c.award_amount, c.cluster_label
            FROM pipeline p JOIN companies c ON c.id = p.company_id
            WHERE p.company_id = ?
            """, (company_id,)
        ) as cur:
            row = await cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Pipeline entry not found")

    data = dict(row)
    years = _calc_years(data.get("date_of_inc"))
    years_str = str(years) if years is not None else "N/A"
    has_gov = "Yes" if (data.get("award_amount") or 0) > 0 else "No"
    location = ", ".join(filter(None, [data.get("city"), data.get("state")])) or "Unknown"

    user_prompt = (
        "Write a short, personalized cold outreach email from a search fund operator "
        "to the owner of this business. Keep it under 150 words. Be genuine, not salesy. "
        "Reference specific details about their business. Do not mention price or acquisition "
        "directly — just express interest in a conversation about the business's future.\n"
        f"Company: {data.get('company_name', 'Unknown')}\n"
        f"Industry: {data.get('industry_sector', 'Unknown')}\n"
        f"Location: {location}\n"
        f"Years in business: {years_str}\n"
        f"Government contracts: {has_gov}\n"
        f"Acquisition score signals: {data.get('cluster_label', 'Unknown')}"
    )

    client = get_anthropic()
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=350,
        system="You are a search fund operator writing personalized, genuine outreach emails to small business owners. Write in first person.",
        messages=[{"role": "user", "content": user_prompt}],
    )

    message = response.content[0].text.strip()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE pipeline SET outreach_message = ? WHERE company_id = ?",
            (message, company_id)
        )
        await db.commit()

    return {"message": message}


@app.post("/search/natural")
async def natural_language_search(payload: NLSearchQuery):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    client = get_anthropic()
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system="You are a search fund acquisition filter parser. Return only valid JSON, no explanation.",
        messages=[{
            "role": "user",
            "content": (
                "Convert this search fund acquisition search query into filter parameters. "
                "Return only valid JSON with these optional fields: "
                "industry (string), state (string, 2-letter code), "
                "min_score (integer 0-100), max_employees (integer), min_years (integer).\n"
                f"Query: {payload.query}"
            ),
        }],
    )

    raw = _strip_code_fences(response.content[0].text)
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

    state_list = _normalize_states(filters.get("state"))
    if state_list:
        placeholders = ",".join("?" * len(state_list))
        conditions.append(f"state IN ({placeholders})")
        params.extend(state_list)

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
    print(f"[NL search] filters={filters}  states={_normalize_states(filters.get('state'))}  SQL={where}  params={params}")

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
