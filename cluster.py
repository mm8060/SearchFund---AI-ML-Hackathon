import subprocess
import sys

try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "scikit-learn"])
    import sklearn

import sqlite3

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DB_PATH = "companies.db"
LABEL_ORDER = ["Prime Target", "Strong Prospect", "Worth Monitoring", "Low Priority"]

SOUTHEAST = {"FL", "GA", "AL", "MS", "TN", "SC", "NC", "VA"}
MIDWEST   = {"OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO"}
SOUTHWEST = {"TX", "AZ", "NM", "CO", "NV"}
NORTHEAST = {"NY", "PA", "NJ", "MA", "CT", "ME", "VT", "NH", "RI"}

HVAC_NAICS     = {"238220", "238210"}
HVAC_SIC       = {1711}
PEST_NAICS     = {"561710"}
PEST_SIC       = {7342}
LANDSCAPE_NAICS= {"561730"}
LANDSCAPE_SIC  = set(range(781, 784))
CONSTRUCT_NAICS= ("236", "237", "238")
CONSTRUCT_SIC  = (1500, 1799)
MFG_NAICS      = ("31", "32", "33")
MFG_SIC        = (2000, 3999)

HVAC_KW      = {"hvac", "heating", "plumbing", "air condition", "pipefitting", "mechanical contractor"}
PEST_KW      = {"pest", "exterminating", "fumigat"}
LANDSCAPE_KW = {"landscap", "lawn", "groundskeep", "irrigation"}
CONSTRUCT_KW = {"construction", "contractor", "roofing", "paving", "masonry", "concrete",
                "drywall", "flooring", "framing", "siding", "carpentry"}
MFG_KW       = {"manufactur", "fabricat", "machining", "metalwork", "welding"}


def encode_industry(naics_code, sic_code, industry_sector):
    n = str(naics_code or "").strip()
    text = (industry_sector or "").lower()

    # NAICS exact / prefix checks
    if n in HVAC_NAICS or n.startswith("2382"):
        return 5
    if n in PEST_NAICS:
        return 4
    if n in LANDSCAPE_NAICS:
        return 3
    if any(n.startswith(p) for p in CONSTRUCT_NAICS):
        return 4
    if any(n.startswith(p) for p in MFG_NAICS):
        return 3

    # SIC checks
    try:
        sic = int(str(sic_code or "").strip())
        if sic in HVAC_SIC:
            return 5
        if sic in PEST_SIC:
            return 4
        if sic in LANDSCAPE_SIC:
            return 3
        if CONSTRUCT_SIC[0] <= sic <= CONSTRUCT_SIC[1]:
            return 4
        if MFG_SIC[0] <= sic <= MFG_SIC[1]:
            return 3
    except ValueError:
        pass

    # Keyword fallback
    if any(kw in text for kw in HVAC_KW):
        return 5
    if any(kw in text for kw in PEST_KW):
        return 4
    if any(kw in text for kw in CONSTRUCT_KW):
        return 4
    if any(kw in text for kw in LANDSCAPE_KW):
        return 3
    if any(kw in text for kw in MFG_KW):
        return 3
    return 1


def encode_region(state):
    s = str(state or "").strip().upper()
    if s in SOUTHEAST:
        return 4
    if s in MIDWEST:
        return 3
    if s in SOUTHWEST:
        return 2
    if s in NORTHEAST:
        return 1
    return 0


def encode_gov_contractor(source):
    return 1.0 if "usaspending" in str(source or "").lower() else 0.0


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(companies)")
    existing_cols = [row[1] for row in cur.fetchall()]
    if "cluster_label" not in existing_cols:
        cur.execute("ALTER TABLE companies ADD COLUMN cluster_label TEXT")
        conn.commit()
        print("Added cluster_label column.")

    cur.execute("""
        SELECT id, acquisition_score,
               naics_code, sic_code, industry_sector, state, source
        FROM companies
        WHERE business_status = 'active'
    """)
    rows = cur.fetchall()
    print(f"Read {len(rows)} active companies.")

    ids = []
    raw = []
    for row in rows:
        raw.append([
            float(row["acquisition_score"] or 0),
            float(encode_industry(row["naics_code"], row["sic_code"], row["industry_sector"])),
            float(encode_region(row["state"])),
            encode_gov_contractor(row["source"]),
        ])
        ids.append(row["id"])

    X = np.array(raw, dtype=float)

    # All 4 features are fully populated; no imputation needed
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # Rank clusters by avg acquisition_score descending, assign labels
    cluster_avg_score = {cid: X[cluster_ids == cid, 0].mean() for cid in range(4)}
    sorted_clusters = sorted(cluster_avg_score, key=cluster_avg_score.get, reverse=True)
    cluster_to_label = {cid: LABEL_ORDER[rank] for rank, cid in enumerate(sorted_clusters)}

    updates = [(cluster_to_label[cid], row_id) for cid, row_id in zip(cluster_ids, ids)]
    conn.executemany("UPDATE companies SET cluster_label = ? WHERE id = ?", updates)
    conn.commit()
    print(f"Saved cluster labels for {len(updates)} companies.\n")

    # Summary table
    print(f"{'='*78}")
    print(f"  CLUSTER SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Cluster':<20} {'Count':>6}  {'Avg Score':>10}  {'Avg Ind':>8}  {'Avg Reg':>8}  {'% GovCon':>9}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}")

    for label in LABEL_ORDER:
        indices = [i for i, cid in enumerate(cluster_ids) if cluster_to_label[cid] == label]
        n = len(indices)
        if n == 0:
            continue
        subset = X[indices]
        print(
            f"  {label:<20} {n:>6}  {subset[:,0].mean():>10.1f}  "
            f"{subset[:,1].mean():>8.2f}  {subset[:,2].mean():>8.2f}  {subset[:,3].mean()*100:>8.1f}%"
        )

    print(f"{'='*78}")
    conn.close()


if __name__ == "__main__":
    main()
