import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = BASE_DIR / "instance" / "users.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(DEFAULT_DB)))

def column_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cur.fetchall())

def table_exists(cur, table):
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cur.fetchone() is not None

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 1) Ensure users table exists (basic login)
    if not table_exists(cur, "users"):
        cur.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
        """)

    # 2) Add experiment-control columns to users
    if not column_exists(cur, "users", "sequence"):
        cur.execute("ALTER TABLE users ADD COLUMN sequence TEXT")
    if not column_exists(cur, "users", "hint_pattern"):
        cur.execute("ALTER TABLE users ADD COLUMN hint_pattern TEXT")
    if not column_exists(cur, "users", "progress"):
        cur.execute("ALTER TABLE users ADD COLUMN progress INTEGER DEFAULT 0")

    # 3) Create responses_v2 (long format)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS responses_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        base_id INTEGER NOT NULL,
        phase TEXT NOT NULL,
        base_desc TEXT,
        descriptors TEXT,
        keywords TEXT,
        reason TEXT,
        concept TEXT,
        time_spent_sec INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

    print("✅ Migration done.")
    print("DB:", DB_PATH)

if __name__ == "__main__":
    main()
