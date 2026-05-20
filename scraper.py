#!/usr/bin/env python3
"""
Search Fund Acquisition Target Data Scraper
============================================
Scrapes ONLY from verified legal US government public sources.

Sources used:
  1. SEC EDGAR (data.sec.gov / www.sec.gov)
     - US federal government (.gov) — Securities Exchange Act public records
     - SEC developer page explicitly authorises programmatic access:
       https://www.sec.gov/developer
     - robots.txt returns HTTP 403 (= no crawl restrictions by RFC 9309 §2.2)
     - Required header: descriptive User-Agent with contact email

  2. USASpending.gov API (api.usaspending.gov)
     - US federal government (.gov) — FFATA-mandated public spending data
       (Federal Funding Accountability and Transparency Act, 31 U.S.C. § 6101)
     - Purpose-built REST API designed for programmatic access:
       https://api.usaspending.gov/docs/
     - robots.txt returns HTTP 404 (= no crawl restrictions)
     - No authentication required; open to all

Sources evaluated and skipped (with reasons):
  - data.sba.gov  : robots.txt Disallow: /api/ blocks the CKAN discovery API
                    needed to find dataset resource IDs.
  - api.census.gov: robots.txt URL returns a WAF hard-block ("Request Rejected");
                    cannot confirm crawl permission.
"""

import csv
import io
import json
import sqlite3
import sys
import time
from datetime import datetime

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = "companies.db"

# SEC requires contact info in User-Agent: https://www.sec.gov/developer
USER_AGENT = "SearchFundPlatform/1.0 (melodyma695@gmail.com)"

SEC_RATE_LIMIT_S  = 0.12   # SEC allows <10 req/sec; we target ~8/sec
USA_RATE_LIMIT_S  = 0.25   # Conservative rate for USASpending API

SOURCES = {
    "SEC_EDGAR": {
        "name": "SEC EDGAR",
        "base_url": "https://data.sec.gov",
        "legal_basis": (
            "US federal government website (.gov). "
            "Data is public record under the Securities Exchange Act of 1934. "
            "SEC explicitly authorises programmatic access with a descriptive User-Agent "
            "at https://www.sec.gov/developer. "
            "robots.txt at data.sec.gov returns HTTP 403, which by RFC 9309 §2.2 means "
            "no crawl restrictions apply."
        ),
        "coverage": "Companies with SEC filing obligations (public and semi-public)",
    },
    "USA_SPENDING": {
        "name": "USASpending.gov (Small Business Awards)",
        "base_url": "https://api.usaspending.gov",
        "legal_basis": (
            "US federal government website (.gov). "
            "Spending data is mandated public under the Federal Funding Accountability "
            "and Transparency Act (FFATA, 31 U.S.C. § 6101). "
            "Purpose-built REST API designed for programmatic access: "
            "https://api.usaspending.gov/docs/ — "
            "robots.txt at api.usaspending.gov returns HTTP 404 (no restrictions)."
        ),
        "coverage": "Companies that received federal contracts (filtered $50K–$10M)",
    },
}

# ── Database schema ───────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS companies (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name     TEXT    NOT NULL,
    industry_sector  TEXT,
    naics_code       TEXT,
    sic_code         TEXT,
    state_of_inc     TEXT,
    date_of_inc      TEXT,
    business_status  TEXT,
    city             TEXT,
    state            TEXT,
    zip_code         TEXT,
    employee_count   TEXT,
    award_amount     REAL,
    business_type    TEXT,
    source           TEXT    NOT NULL,
    source_id        TEXT,
    raw_data         TEXT,
    scraped_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_name, city, state, source)
);

CREATE INDEX IF NOT EXISTS idx_companies_state  ON companies(state);
CREATE INDEX IF NOT EXISTS idx_companies_naics  ON companies(naics_code);
CREATE INDEX IF NOT EXISTS idx_companies_source ON companies(source);
CREATE INDEX IF NOT EXISTS idx_companies_status ON companies(business_status);
CREATE INDEX IF NOT EXISTS idx_companies_name   ON companies(company_name);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT,
    records      INTEGER,
    started_at   TEXT,
    finished_at  TEXT,
    status       TEXT,
    notes        TEXT
);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_batch(conn: sqlite3.Connection, rows: list) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR IGNORE INTO companies
          (company_name, industry_sector, naics_code, sic_code,
           state_of_inc, date_of_inc, business_status,
           city, state, zip_code, employee_count, award_amount, business_type,
           source, source_id, raw_data)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})


def http_get(url: str, timeout: int = 30, **kwargs) -> requests.Response:
    resp = SESSION.get(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp


def http_post(url: str, timeout: int = 30, **kwargs) -> requests.Response:
    resp = SESSION.post(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp


# ── Source 1: SEC EDGAR ───────────────────────────────────────────────────────

def scrape_sec_edgar(conn: sqlite3.Connection, limit: int = 300, start: int = 3000) -> int:
    """
    1. Downloads the full EDGAR company list (one request).
    2. Fetches per-company submission metadata (rate-limited).

    Fields captured: company name, SIC code, SIC description (industry),
    state of incorporation, entity type, city, state, zip, estimated active status.

    API docs: https://www.sec.gov/developer
    """
    src = SOURCES["SEC_EDGAR"]
    started = datetime.now().isoformat()
    print(f"\n{'='*64}")
    print(f"SOURCE 1 : {src['name']}")
    print(f"URL      : {src['base_url']}")
    print(f"LEGAL    : {src['legal_basis']}")
    print(f"COVERAGE : {src['coverage']}")
    print(f"{'='*64}")

    # ── Step 1: full company list (single lightweight request) ──
    print("\n  Fetching master company list from SEC EDGAR…")
    resp = http_get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"Host": "www.sec.gov"},
    )
    tickers: dict = resp.json()
    total_known = len(tickers)
    print(f"  → {total_known:,} companies registered in EDGAR")

    # Skip the first `start` entries — those tend to be mega-caps with low CIK numbers.
    # Companies past position 3000 are smaller/less-followed firms.
    companies = list(tickers.values())[start : start + limit]
    print(f"  → Fetching details for entries {start}–{start+len(companies)} "
          f"(rate-limited to ~8 req/sec)…\n")

    # ── Step 2: per-company submission detail ──
    records, errors = 0, 0
    batch: list = []

    for i, company in enumerate(companies):
        cik = str(company["cik_str"]).zfill(10)
        time.sleep(SEC_RATE_LIMIT_S)

        try:
            data = http_get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers={"Host": "data.sec.gov"},
                timeout=15,
            ).json()

            name = (data.get("name") or "").strip()
            if not name:
                continue

            sic_code    = str(data.get("sic") or "")
            sic_desc    = data.get("sicDescription") or ""
            state_inc   = (data.get("stateOfIncorporationDescription")
                           or data.get("stateOfIncorporation") or "")
            entity_type = data.get("entityType") or ""

            addrs    = data.get("addresses", {})
            biz_addr = addrs.get("business", {})
            city     = biz_addr.get("city") or ""
            state    = biz_addr.get("stateOrCountry") or ""
            zip_code = biz_addr.get("zipCode") or ""

            dates = (data.get("filings", {})
                         .get("recent", {})
                         .get("filingDate", []))
            if dates:
                last_filing = dates[0]
                status = "active" if last_filing >= "2022-01-01" else "possibly inactive"
            else:
                status = "unknown"

            batch.append((
                name,
                sic_desc or None,
                None,
                sic_code or None,
                state_inc or None,
                None,
                status,
                city or None,
                state or None,
                zip_code or None,
                None,
                None,
                entity_type or None,
                src["name"],
                cik,
                json.dumps({
                    "cik": cik,
                    "ticker": company.get("ticker", ""),
                    "last_filing": dates[0] if dates else None,
                    "fiscal_year_end": data.get("fiscalYearEnd"),
                }),
            ))
            records += 1

            if len(batch) >= 100:
                insert_batch(conn, batch)
                batch.clear()

            if (i + 1) % 50 == 0 or i + 1 == len(companies):
                sys.stdout.write(f"\r  [{i+1}/{len(companies)}] {records} records stored…")
                sys.stdout.flush()

        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"\n  Warning: CIK {cik} → {exc}")

    insert_batch(conn, batch)
    sys.stdout.write(f"\r  [{len(companies)}/{len(companies)}] {records} records stored… done.\n")

    conn.execute(
        "INSERT INTO scrape_runs VALUES (NULL,?,?,?,?,?,?)",
        (src["name"], records, started, datetime.now().isoformat(),
         "success", f"{records} companies; {errors} errors"),
    )
    conn.commit()
    print(f"  ✓ {records} companies stored from SEC EDGAR  ({errors} fetch errors)")
    return records


# ── Source 2: USASpending.gov ─────────────────────────────────────────────────

_SEARCH_URL  = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
_DETAIL_URL  = "https://api.usaspending.gov/api/v2/awards/{award_id}/"

_SEARCH_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Place of Performance City Name",
    "Place of Performance State Code",
    "Place of Performance Zip5",
    "Award Amount",
    "Award Date",
    "Description",
]


def _fetch_award_naics(generated_id: str) -> tuple:
    """Return (naics_code, naics_description) from award detail API."""
    try:
        data = http_get(
            _DETAIL_URL.format(award_id=generated_id),
            timeout=15,
        ).json()
        tx = data.get("latest_transaction_contract_data") or {}
        return tx.get("naics") or "", tx.get("naics_description") or ""
    except Exception:
        return "", ""


def scrape_usaspending(
    conn: sqlite3.Connection,
    pages: int = 30,
    enrich_naics: bool = True,
    naics_sample: int = 500,
) -> int:
    """
    Queries USASpending.gov for federal contracts in the $50K–$10M range
    (awards typical of small businesses).

    Per-award NAICS codes are fetched from the detail endpoint (up to
    `naics_sample` awards) because the search endpoint does not return them
    in the aggregated award view.

    API: https://api.usaspending.gov/api/v2/search/spending_by_award/
    """
    src = SOURCES["USA_SPENDING"]
    started = datetime.now().isoformat()
    print(f"\n{'='*64}")
    print(f"SOURCE 2 : {src['name']}")
    print(f"URL      : {src['base_url']}")
    print(f"LEGAL    : {src['legal_basis']}")
    print(f"COVERAGE : {src['coverage']}")
    print(f"{'='*64}\n")

    records, errors = 0, 0
    # Accumulate all awards first, then enrich with NAICS
    all_awards: list[dict] = []

    for page in range(1, pages + 1):
        time.sleep(USA_RATE_LIMIT_S)

        payload = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                # $50K–$2M range + ascending sort targets contracts
                # more likely issued to small / mid-size businesses.
                "award_amounts": [{"lower_bound": 50_000, "upper_bound": 2_000_000}],
            },
            "fields": _SEARCH_FIELDS,
            "sort": "Award Amount",
            "order": "asc",
            "page": page,
            "limit": 100,
            "subawards": False,
        }

        try:
            resp = http_post(_SEARCH_URL, json=payload)
            data = resp.json()
        except Exception as exc:
            errors += 1
            print(f"  Warning: page {page} failed → {exc}")
            if errors > 5:
                break
            continue

        results = data.get("results", [])
        if not results:
            print(f"  No more results after page {page - 1}.")
            break

        all_awards.extend(results)
        meta = data.get("page_metadata", {})
        sys.stdout.write(
            f"\r  Fetched page {page}/{pages}: {len(all_awards)} awards total…"
        )
        sys.stdout.flush()

        if not meta.get("hasNext", True):
            break

    print(f"\n  Collected {len(all_awards)} awards from search API")

    # ── Optionally enrich a sample with NAICS from detail API ──
    naics_map: dict[str, tuple] = {}  # generated_id → (code, desc)
    if enrich_naics and all_awards:
        sample = all_awards[:naics_sample]
        print(f"  Fetching NAICS codes for {len(sample)} awards via detail API…")
        for j, award in enumerate(sample):
            gen_id = award.get("generated_internal_id", "")
            if not gen_id:
                continue
            time.sleep(USA_RATE_LIMIT_S)
            code, desc = _fetch_award_naics(gen_id)
            naics_map[gen_id] = (code, desc)

            if (j + 1) % 50 == 0 or j + 1 == len(sample):
                hits = sum(1 for c, _ in naics_map.values() if c)
                sys.stdout.write(
                    f"\r  NAICS enrichment: {j+1}/{len(sample)} "
                    f"({hits} codes found)…"
                )
                sys.stdout.flush()
        print()

    # ── Insert into database ──
    batch: list = []
    for award in all_awards:
        name = (award.get("Recipient Name") or "").strip()
        if not name or name.upper() in ("MULTIPLE RECIPIENTS", "VARIOUS"):
            continue

        gen_id    = award.get("generated_internal_id", "")
        award_id  = award.get("Award ID", "")
        city      = award.get("Place of Performance City Name") or ""
        state     = award.get("Place of Performance State Code") or ""
        zip_code  = award.get("Place of Performance Zip5") or ""
        desc      = award.get("Description") or ""
        date_val  = award.get("Award Date") or ""
        amount    = award.get("Award Amount")

        naics_code, naics_desc = naics_map.get(gen_id, ("", ""))
        # Fall back to description as a rough industry proxy
        industry = naics_desc or (desc[:100] if desc and len(desc) > 3 else None)

        try:
            award_amount = float(amount) if amount is not None else None
        except (ValueError, TypeError):
            award_amount = None

        batch.append((
            name,
            industry or None,
            naics_code or None,
            None,
            None,
            date_val or None,
            "active",
            city or None,
            state or None,
            zip_code or None,
            None,
            award_amount,
            None,
            src["name"],
            award_id or None,
            json.dumps({
                "award_id": award_id,
                "generated_id": gen_id,
                "award_amount": amount,
                "award_date": date_val,
                "description": desc[:200] if desc else None,
            }),
        ))
        records += 1

        if len(batch) >= 500:
            insert_batch(conn, batch)
            batch.clear()

    insert_batch(conn, batch)

    conn.execute(
        "INSERT INTO scrape_runs VALUES (NULL,?,?,?,?,?,?)",
        (src["name"], records, started, datetime.now().isoformat(),
         "success",
         f"{records} award recipients; NAICS enriched for "
         f"{sum(1 for c,_ in naics_map.values() if c)} of "
         f"{len(naics_map)} awards; {errors} page errors"),
    )
    conn.commit()
    print(f"  ✓ {records} companies stored from USASpending.gov")
    return records


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(conn: sqlite3.Connection) -> None:
    print(f"\n{'='*64}")
    print("DATABASE SUMMARY")
    print(f"{'='*64}")

    total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    print(f"\nTotal companies stored : {total:,}")

    print("\nBy source:")
    for src, n in conn.execute(
        "SELECT source, COUNT(*) FROM companies GROUP BY source ORDER BY COUNT(*) DESC"
    ):
        print(f"  {src:<50} {n:>8,}")

    print("\nTop 12 states (location of business):")
    for state, n in conn.execute(
        """SELECT state, COUNT(*) n FROM companies
           WHERE state IS NOT NULL AND state != ''
           GROUP BY state ORDER BY n DESC LIMIT 12"""
    ):
        print(f"  {state:<6} {n:>8,}")

    print("\nTop 10 industries — NAICS (USASpending data):")
    for sector, n in conn.execute(
        """SELECT industry_sector, COUNT(*) n FROM companies
           WHERE source LIKE '%USASpending%'
             AND industry_sector IS NOT NULL AND industry_sector != ''
           GROUP BY industry_sector ORDER BY n DESC LIMIT 10"""
    ):
        print(f"  {n:>6,}  {str(sector)[:65]}")

    print("\nTop 10 industries — SIC (SEC EDGAR data):")
    for sector, n in conn.execute(
        """SELECT industry_sector, COUNT(*) n FROM companies
           WHERE source LIKE '%EDGAR%'
             AND industry_sector IS NOT NULL AND industry_sector != ''
           GROUP BY industry_sector ORDER BY n DESC LIMIT 10"""
    ):
        print(f"  {n:>6,}  {str(sector)[:65]}")

    print("\nSample records (10 — mixed sources):")
    header = f"  {'Company':<38} {'City':<16} {'St':<4} {'Industry':<32} Source"
    print(header)
    print("  " + "-"*110)
    for row in conn.execute(
        """SELECT company_name, city, state, industry_sector, source
           FROM companies ORDER BY RANDOM() LIMIT 10"""
    ):
        print(
            f"  {str(row[0]):<38.38} {str(row[1] or ''):<16.16} "
            f"{str(row[2] or ''):<4.4} {str(row[3] or ''):<32.32} {str(row[4])[:20]}"
        )


# ── Skipped-sources report ────────────────────────────────────────────────────

def print_skipped_sources() -> None:
    print(f"\n{'='*64}")
    print("SOURCES EVALUATED BUT SKIPPED")
    print(f"{'='*64}")
    skipped = [
        {
            "name": "data.sba.gov (SBA Open Data / CKAN API)",
            "reason": (
                "robots.txt explicitly Disallow: /api/ — this path covers the CKAN "
                "API used to discover dataset resource IDs. Without the API there is "
                "no reliable programmatic way to find download URLs. Skipped per your "
                "strict-legality requirement."
            ),
        },
        {
            "name": "api.census.gov (US Census Bureau API)",
            "reason": (
                "Fetching robots.txt returns a WAF hard-block ('Request Rejected'). "
                "Cannot confirm crawl permission is granted. Skipped."
            ),
        },
    ]
    for s in skipped:
        print(f"\n  Source : {s['name']}")
        print(f"  Reason : {s['reason']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("="*64)
    print("Search Fund Acquisition Target Scraper")
    print("Legal sources: US Government public data only")
    print(f"Database     : {DB_PATH}")
    print(f"Started      : {datetime.now().isoformat(timespec='seconds')}")
    print("="*64)

    conn = open_db(DB_PATH)

    total = 0
    total += scrape_sec_edgar(conn, limit=300, start=3000)
    total += scrape_usaspending(conn, pages=30, enrich_naics=True, naics_sample=300)

    print_summary(conn)
    print_skipped_sources()

    conn.close()
    print(f"\n✓ Complete. {total:,} records written to {DB_PATH}")


if __name__ == "__main__":
    main()
