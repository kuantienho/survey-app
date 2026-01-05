import sqlite3

# 建立資料庫連線
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# 建立 users 表格（如果還沒建立）
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
''')

# 固定帳號密碼：從 user000 ~ user050，密碼為 pass000 ~ pass050
user_list = []

for i in range(0, 51):  # 0 到 50 共 51 筆
    username = f"user{i:03}"   # user000 ~ user050
    password = f"pass{i:03}"   # 對應的密碼 pass000 ~ pass050
    
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        user_list.append((username, password))
    except sqlite3.IntegrityError:
        print(f"{username} 已存在，略過")

# 儲存變更
conn.commit()
conn.close()

# 印出所有帳號密碼
print("✅ 以下為固定帳號密碼（共 51 組）")
for u, p in user_list:
    print(f"{u} / {p}")
