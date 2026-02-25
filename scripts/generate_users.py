import os
import sqlite3
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = BASE_DIR / "instance" / "users.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(DEFAULT_DB)))

def ensure_users_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        sequence TEXT,
        hint_pattern TEXT,
        progress INTEGER DEFAULT 0
    )
    """)
    conn.commit()

def main(prefix="A", start=1, count=30, width=2, out_csv="user_credentials.csv"):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    ensure_users_table(conn)
    cur = conn.cursor()

    created = []
    for i in range(start, start + count):
        username = f"USER{prefix}{i:0{width}d}"
        password = f"PASS{prefix}{i:0{width}d}"

        try:
            cur.execute(
                "INSERT INTO users (username, password, progress) VALUES (?, ?, 0)",
                (username, password)
            )
            created.append((username, password))
        except sqlite3.IntegrityError:
            continue

    conn.commit()
    conn.close()

    # CSV 永遠輸出到「本專案 scripts/」
    out_path = BASE_DIR / "scripts" / out_csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["username", "password"])
        w.writerows(created)

    print(f"Created {len(created)} users.")
    print("DB :", DB_PATH)
    print("CSV:", out_path)

if __name__ == "__main__":
    main(prefix="A", start=1, count=30, width=2)
