import sqlite3
from datetime import datetime

DB_PATH = "companies.db"

# NAICS prefixes for target industries (stable, boring, acquirable businesses)
TARGET_NAICS = {
    "238220": 30,  # Plumbing, Heating, Air-Conditioning
    "238210": 30,  # Electrical Contractors
    "238110": 30,  # Poured Concrete Foundation
    "238130": 30,  # Framing Contractors
    "238160": 30,  # Roofing Contractors
    "238170": 30,  # Siding Contractors
    "238310": 30,  # Drywall and Insulation
    "238320": 30,  # Painting and Wall Covering
    "238330": 30,  # Flooring Contractors
    "238350": 30,  # Finish Carpentry
    "238390": 30,  # Other Building Finishing
    "238910": 30,  # Site Preparation Contractors
    "238990": 30,  # All Other Specialty Trade
    "561730": 30,  # Landscaping Services
    "561710": 30,  # Exterminating and Pest Control
    "562111": 30,  # Solid Waste Collection
    "562119": 30,  # Other Waste Collection
    "562212": 30,  # Solid Waste Landfill
    "811111": 30,  # General Automotive Repair
    "811113": 30,  # Automotive Transmission Repair
    "811118": 30,  # Other Automotive Mechanical Repair
    "811121": 30,  # Automotive Body, Paint, Interior
    "811192": 30,  # Car Washes
    "811310": 30,  # Commercial/Industrial Machinery Repair
    "811412": 30,  # Appliance Repair and Maintenance
}

TARGET_NAICS_PREFIXES = [
    ("236", 25),   # Construction of Buildings
    ("237", 25),   # Heavy and Civil Engineering Construction
    ("238", 25),   # Specialty Trade Contractors
    ("31",  22),   # Manufacturing
    ("32",  22),   # Manufacturing
    ("33",  22),   # Manufacturing
    ("562", 25),   # Waste Management
    ("811", 25),   # Repair and Maintenance
    ("561", 18),   # Administrative/Support (landscaping, pest, etc.)
    ("484", 20),   # Trucking
    ("423", 18),   # Merchant Wholesalers, Durable Goods
    ("424", 18),   # Merchant Wholesalers, Nondurable Goods
    ("441", 15),   # Motor Vehicle Dealers
    ("452", 12),   # General Merchandise Stores
]

UNFAVORABLE_NAICS_PREFIXES = [
    ("52",  2),    # Finance and Insurance
    ("53",  5),    # Real Estate
    ("54",  5),    # Professional, Scientific, Technical
    ("61",  5),    # Educational Services
    ("62",  8),    # Health Care (some OK, but complex)
    ("71",  8),    # Arts, Entertainment
    ("92",  3),    # Public Administration
    ("51",  5),    # Information/Tech
]

# SIC code ranges for target industries
TARGET_SIC_RANGES = [
    (1500, 1799, 28),   # Construction
    (2000, 3999, 22),   # Manufacturing
    (4953, 4953, 28),   # Refuse Systems
    (7342, 7342, 28),   # Disinfecting/Pest Control
    (7500, 7549, 28),   # Auto Repair/Services
    (7690, 7699, 25),   # Misc Repair Services
    (1711, 1711, 30),   # Plumbing, Heating, Air-Conditioning
    (781,  783,  30),   # Landscaping/Lawn/Garden
    (4210, 4215, 20),   # Trucking & Warehousing
    (5080, 5099, 18),   # Industrial Machinery Wholesale
]

UNFAVORABLE_SIC_RANGES = [
    (6000, 6799, 2),    # Finance, Insurance, Real Estate
    (7370, 7379, 5),    # Computer Services
    (8000, 8099, 8),    # Health Services
    (8200, 8299, 5),    # Educational Services
    (9000, 9999, 3),    # Government
]

TARGET_KEYWORDS = [
    "hvac", "heating", "cooling", "air condition", "plumb", "pipefitting",
    "landscap", "lawn", "groundskeep", "pest control", "exterminating",
    "roofing", "paving", "asphalt", "concrete", "masonry", "flooring",
    "painting", "drywall", "insulation", "carpentry", "siding",
    "construction", "contractor", "remodeling", "renovation",
    "manufacturing", "fabricat", "machining", "welding", "metalwork",
    "auto repair", "automotive", "auto service", "car wash", "body shop",
    "waste", "refuse", "sanitation", "recycling", "hauling",
    "trucking", "freight", "delivery", "logistics",
    "janitorial", "cleaning service", "custodial",
    "electrical contractor", "electrical service",
    "appliance repair", "equipment repair",
    "pool service", "irrigation",
]

UNFAVORABLE_KEYWORDS = [
    "pharma", "biotech", "software", "saas", "technology",
    "investment", "fund", "hedge", "venture", "capital",
    "bank", "financial", "insurance", "mortgage", "securities",
    "university", "education", "school", "college",
]


def score_industry(naics_code, sic_code, industry_sector):
    text = (industry_sector or "").lower()

    if naics_code:
        code = str(naics_code).strip()
        if code in TARGET_NAICS:
            return TARGET_NAICS[code]
        for prefix, pts in TARGET_NAICS_PREFIXES:
            if code.startswith(prefix):
                return pts
        for prefix, pts in UNFAVORABLE_NAICS_PREFIXES:
            if code.startswith(prefix):
                return pts
        return 10  # known NAICS but uncategorized

    if sic_code:
        try:
            sic = int(str(sic_code).strip())
        except ValueError:
            sic = None
        if sic is not None:
            for lo, hi, pts in TARGET_SIC_RANGES:
                if lo <= sic <= hi:
                    return pts
            for lo, hi, pts in UNFAVORABLE_SIC_RANGES:
                if lo <= sic <= hi:
                    return pts
            return 10

    # Fall back to keyword matching on industry_sector text
    if text:
        for kw in UNFAVORABLE_KEYWORDS:
            if kw in text:
                return 3
        for kw in TARGET_KEYWORDS:
            if kw in text:
                return 22
        return 8  # has sector text but no keyword match

    return 0  # no industry data at all


def score_revenue(award_amount):
    if award_amount is None:
        return 0
    amt = float(award_amount)
    if 500_000 <= amt <= 10_000_000:
        return 25
    if 100_000 <= amt < 500_000:
        return 18
    if 10_000_000 < amt <= 50_000_000:
        return 12
    if amt < 100_000:
        return 5
    return 0  # > $50M


def score_age(date_of_inc):
    if not date_of_inc:
        return 0
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y"):
        try:
            inc_date = datetime.strptime(str(date_of_inc).strip(), fmt)
            years = (datetime.now() - inc_date).days / 365.25
            if years >= 20:
                return 25
            if years >= 10:
                return 20
            if years >= 5:
                return 10
            return 0
        except ValueError:
            continue
    return 0


def score_employees(employee_count):
    if employee_count is None:
        return 0
    try:
        count = int(str(employee_count).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0
    if 10 <= count <= 75:
        return 20
    if 5 <= count < 10 or 75 < count <= 150:
        return 12
    if 1 <= count < 5 or 150 < count <= 500:
        return 5
    return 0  # > 500 (too large) or 0


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        ALTER TABLE companies ADD COLUMN acquisition_score INTEGER
    """)
    conn.commit()
    print("Added acquisition_score column.")
    cur2 = conn.cursor()

    cur.execute("""
        SELECT id, naics_code, sic_code, industry_sector,
               award_amount, date_of_inc, employee_count
        FROM companies
        WHERE business_status = 'active'
    """)
    rows = cur.fetchall()
    print(f"Scoring {len(rows)} active companies...")

    updates = []
    for row in rows:
        s_industry  = score_industry(row["naics_code"], row["sic_code"], row["industry_sector"])
        s_revenue   = score_revenue(row["award_amount"])
        s_age       = score_age(row["date_of_inc"])
        s_employees = score_employees(row["employee_count"])
        total = min(100, s_industry + s_revenue + s_age + s_employees)
        updates.append((total, row["id"]))

    cur2.executemany(
        "UPDATE companies SET acquisition_score = ? WHERE id = ?",
        updates
    )
    conn.commit()

    # Summary
    cur.execute("SELECT acquisition_score FROM companies WHERE business_status = 'active'")
    scores = [r[0] for r in cur.fetchall() if r[0] is not None]

    avg = sum(scores) / len(scores) if scores else 0
    buckets = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for s in scores:
        if s <= 25:
            buckets["0-25"] += 1
        elif s <= 50:
            buckets["26-50"] += 1
        elif s <= 75:
            buckets["51-75"] += 1
        else:
            buckets["76-100"] += 1

    cur.execute("""
        SELECT company_name, industry_sector, naics_code, sic_code,
               award_amount, acquisition_score
        FROM companies
        WHERE business_status = 'active'
        ORDER BY acquisition_score DESC
        LIMIT 10
    """)
    top10 = cur.fetchall()

    conn.close()

    print(f"\n{'='*60}")
    print(f"  ACQUISITION SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"  Companies scored : {len(scores)}")
    print(f"  Average score    : {avg:.1f} / 100")
    print(f"\n  Score distribution:")
    for bucket, count in buckets.items():
        pct = count / len(scores) * 100 if scores else 0
        bar = "#" * int(pct / 2)
        print(f"    {bucket:>7}  {count:5d}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Top 10 acquisition targets:")
    print(f"  {'#':<3} {'Score':>5}  {'Company':<40}  {'Industry/Code'}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*40}  {'-'*20}")
    for i, row in enumerate(top10, 1):
        name = (row[0] or "Unknown")[:40]
        industry = row[1] or row[2] or row[3] or "N/A"
        industry = str(industry)[:22]
        score = row[5]
        award = f"${row[4]:,.0f}" if row[4] else "N/A"
        print(f"  {i:<3} {score:>5}  {name:<40}  {industry}  (award: {award})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
