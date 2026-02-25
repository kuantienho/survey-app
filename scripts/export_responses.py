import os
import sqlite3
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = BASE_DIR / "instance" / "users.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(DEFAULT_DB)))

def main(out_path=None):
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = BASE_DIR / "exports" / f"responses_{ts}.csv"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Join users to get username
    cur.execute("""
        SELECT
            r.id,
            r.user_id,
            u.username,
            r.base_id,
            r.phase,
            r.base_desc,
            r.descriptors,
            r.keywords,
            r.reason,
            r.concept,
            r.time_spent_sec,
            r.created_at
        FROM responses_v2 r
        JOIN users u ON u.id = r.user_id
        ORDER BY r.created_at ASC, r.id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    headers = [
        "id","user_id","username","base_id","phase",
        "base_desc","descriptors","keywords","reason","concept",
        "time_spent_sec","created_at"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r[h] for h in headers])

    print(f"✅ Exported {len(rows)} rows")
    print("CSV:", out_path)
    print("DB :", DB_PATH)

if __name__ == "__main__":
    main()
