from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_me")

# --- DB  ---
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# --- Login ---
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()

        if user:
            session["user_id"] = user["id"]
            # 登入成功 → 指導頁 Page 1
            return redirect(url_for("instructions"))

        return "Invalid credentials or Wrong Password", 401

    return render_template("login.html")

# --- Instructions Page 1 ---
@app.route("/instructions", methods=["GET"])
def instructions():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("instructions.html")

# --- Instructions Page 2 ---
# 允許 GET 與 POST：若你在 instructions.html 用 <form method="post" action="{{ url_for('instructions2') }}">
@app.route("/instructions2", methods=["GET", "POST"])
def instructions2():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        # 這裡如果未來要處理表單資料可加上邏輯
        # 目前僅顯示同一頁即可（不做其他跳轉）
        pass

    return render_template("instructions2.html")

@app.route("/base1", methods=["GET", "POST"])
def base1():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        # 這裡如果未來要處理表單資料可加上邏輯
        # 目前僅顯示同一頁即可（不做其他跳轉）
        pass

    return render_template("base1.html")

@app.route("/base2", methods=["GET", "POST"])
def base2():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        # TODO: save form data (Umbrella)
        return redirect(url_for("base2"))
    return render_template("base2.html")

@app.route("/base3", methods=["GET", "POST"])
def base3():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        # TODO: save form data (Chair)
        return redirect(url_for("base3"))
    return render_template("base3.html")

@app.route("/base4", methods=["GET", "POST"])
def base4():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        # TODO: save form data (Power Supply)
        return redirect(url_for("base4"))
    return render_template("base4.html")



# --- Run ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)
