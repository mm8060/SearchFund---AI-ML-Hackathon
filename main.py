from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = "companies.db"


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
                   award_amount, business_type, source, acquisition_score
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

        # Verify company exists
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
