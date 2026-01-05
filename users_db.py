import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Users table
c.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
''')

# Responses table
c.execute('''
CREATE TABLE responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    answer1 TEXT,
    answer2 TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')

# Add test user
c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("testuser", "testpass"))

conn.commit()
conn.close()
