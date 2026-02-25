from flask import (
    Flask, render_template, request, redirect,
    url_for, session, Response, abort
)
import sqlite3
import os
import csv
from io import StringIO

from hint_engine.pen_tool import get_pen_add
from hint_engine.psu_tool import get_psu_add
from hint_engine.chair_tool import get_chair_add
from hint_engine.umbrella_tool import get_umbrella_add


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_me")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 仍保留你原本的預設：instance/users.db
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "instance", "users.db")
DB_PATH = os.environ.get("DB_PATH", DEFAULT_DB_PATH)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin_change_me")

# =========================================================
# META + HINT DISPATCH
# =========================================================

BASES_META = {
    1: {
        "en_name": "Pen",
        "ja_name": "ペン",
        "en_desc": (
            "Pen-type task: any similar writing tool is acceptable (pen / marker / pencil, etc.). "
            "Choose ONE object you want to work with and generate an idea or design."
        ),
        "ja_desc": (
            "ペン系課題：ペン／マーカー／鉛筆など、類似する筆記具はすべて「ペン」として扱い、提出OKです。"
            "ここで扱いたい対象を1つ選び、アイデア／デザインを考えてください。"
        ),
        "ph_base_desc": (
            "Example: A pen / marker pen / pencil is a tool used to write or draw. "
            "It usually has a body, a tip, and ink or lead, and is held in the hand.\n"
            "Now describe the object YOU choose to use for ideation (what it is, how it works, and where it is used)."
        ),
        "ph_base_desc_ja": (
            "例：ペン／マーカーペン／鉛筆は、文字を書いたり絵を描いたりするための道具です。"
            "通常、本体、先端、インクまたは芯で構成され、手で持って使用します。\n"
            "次に、あなたがアイデア出しに使用する対象について説明してください（それが何か、どのように機能し、どこで使われるか）。"
        ),
        "ph_descriptors": (
            "Describe problems of the object you chose in under 3 sentences. "
            "Focus on dissatisfaction or pain points. "
            "Avoid praise or positive descriptions."
        ),
        "ph_descriptors_ja": (
            "選んだ対象の問題点を3文以内で記述してください。"
            "不満点や使いにくさなどの課題に焦点を当ててください。"
            "良い点やポジティブな表現は避けてください。"
        ),
        "ph_descriptors_example": (
            "Ink smudges easily when writing fast.\n"
            "Grip becomes slippery after long use.\n"
            "Line thickness is inconsistent."
        ),
        "ph_descriptors_example_ja": (
            "速く書くとインクがにじみやすい。\n"
            "長時間使用するとグリップが滑りやすくなる。\n"
            "線の太さが安定しない。"
        ),
        "ph_concept": (
            "Based on the problems you described, propose an idea or design to improve or solve them (3–6 sentences). "
            "If additives are shown, use at least ONE additive and explain what feature inspired you and how it relates to your idea. "
            "Your submission does not have to be the same object as the base (related designs such as a pencil case or other stationery are acceptable)."
        ),
        "ph_concept_ja": (
            "記述した問題点をもとに、それらを改善または解決するためのアイデアやデザインを3〜6文で提案してください。"
            "Additiveが提示されている場合は、少なくとも1つを使用し、どの特徴に着目し、それがどのようにアイデアと関係しているかを説明してください。"
            "提出物は必ずしも元の対象と同じである必要はなく、ペンケースなど関連する文房具の提案でも構いません。"
        ),
        "ph_concept_example": (
            "To address smudging and grip issues, I propose a writing tool system focused on controlled ink flow and tactile feedback. "
            "Inspired by the additive feature of micro-textured surfaces, the grip design improves stability without increasing thickness. "
            "Instead of redesigning the pen itself, this idea could take the form of a modular writing sleeve compatible with pens, pencils, or markers."
        ),
        "ph_concept_example_ja": (
            "インクのにじみやグリップの問題に対応するため、インク制御と触覚フィードバックに着目した筆記具システムを提案します。"
            "Additiveの特徴である微細なテクスチャ構造に着想を得て、太さを変えずに安定性を高めるグリップデザインを採用します。"
            "ペン本体を作り替えるのではなく、ペン・鉛筆・マーカーに対応するモジュール式スリーブとして展開することも可能です。"
        ),
        "visual": "visuals/all_pen.png",
    },

    2: {
        "en_name": "Power Supply",
        "ja_name": "電源装置",
        "en_desc": (
            "Power-supply-type task: similar products are acceptable (charger / AC adapter / power strip / battery pack, etc.). "
            "Choose ONE object you want to work with and generate an idea or design."
        ),
        "ja_desc": (
            "電源系課題：充電器／ACアダプタ／電源タップ／バッテリーパックなど、類似する製品はすべて「電源装置」として扱い、提出OKです。"
            "ここで扱いたい対象を1つ選び、アイデア／デザインを考えてください。"
        ),
        "ph_base_desc": (
            "Example: A power supply or charger is a device that provides electrical power to other devices. "
            "It may include plugs, cables, ports, and protection components.\n"
            "Now describe the object YOU choose (what it powers, how it connects, and where it is used)."
        ),
        "ph_base_desc_ja": (
            "例：電源装置や充電器は、他の機器に電力を供給するための装置です。"
            "プラグ、ケーブル、ポート、保護回路などで構成される場合があります。\n"
            "次に、あなたが選んだ対象について説明してください（何に電力を供給するか、どのように接続され、どこで使われるか）。"
        ),
        "ph_descriptors": (
            "Describe problems of the object you chose in under 3 sentences. "
            "Focus on dissatisfaction or pain points. "
            "Avoid praise or positive descriptions."
        ),
        "ph_descriptors_ja": (
            "選んだ対象の問題点を3文以内で記述してください。"
            "不満点や使いにくさなどの課題に焦点を当ててください。"
            "良い点やポジティブな表現は避けてください。"
        ),
        "ph_descriptors_example": (
            "Gets hot during extended use.\n"
            "Cables are difficult to organize.\n"
            "Status indicators are unclear."
        ),
        "ph_descriptors_example_ja": (
            "長時間使用すると熱くなる。\n"
            "ケーブルの整理がしにくい。\n"
            "状態表示が分かりにくい。"
        ),
        "ph_concept": (
            "Based on the problems you described, propose an idea or design to improve or solve them (3–6 sentences). "
            "If additives are shown, use at least ONE additive and explain what feature inspired you and how it relates to your idea. "
            "Your submission can be a related solution or system, not necessarily the same device."
        ),
        "ph_concept_ja": (
            "記述した問題点をもとに、それらを改善または解決するためのアイデアやデザインを3〜6文で提案してください。"
            "Additiveが提示されている場合は、少なくとも1つを使用し、その特徴がどのようにアイデアに反映されているかを説明してください。"
            "提出物は必ずしも同一の装置である必要はなく、関連するシステム提案でも構いません。"
        ),
        "ph_concept_example": (
            "Based on overheating and cable clutter, I propose a power management dock that separates heat-generating components from user interaction areas. "
            "Inspired by the additive feature of thermal zoning, the design visually and physically divides power conversion and cable management. "
            "Rather than a single charger, this system functions as a shared desktop power station with clearer feedback."
        ),
        "ph_concept_example_ja": (
            "発熱やケーブルの混雑といった問題を踏まえ、熱を発生する部分と操作エリアを分離した電源管理ドックを提案します。"
            "Additiveの特徴であるサーマルゾーニングに着想を得て、電力変換部とケーブル管理部を視覚的・物理的に分けた構成とします。"
            "単体の充電器ではなく、明確なフィードバックを持つデスクトップ用電源ステーションとして機能します。"
        ),
        "visual": "visuals/all_psu.png",
    },

    3: {
        "en_name": "Chair",
        "ja_name": "椅子",
        "en_desc": (
            "Chair-type task: similar seating is acceptable (stool / office chair / bench / floor chair, etc.). "
            "Choose ONE object you want to work with and generate an idea or design."
        ),
        "ja_desc": (
            "椅子系課題：スツール／オフィスチェア／ベンチ／座椅子など、類似する座るための製品はすべて「椅子」として扱い、提出OKです。"
            "ここで扱いたい対象を1つ選び、アイデア／デザインを考えてください。"
        ),
        "ph_base_desc": (
            "Example: A chair or stool is furniture designed to support the body while sitting. "
            "It may include legs, a seat, a backrest, and materials affecting comfort.\n"
            "Now describe the object YOU choose (posture, user, environment, and main structure)."
        ),
        "ph_base_desc_ja": (
            "例：椅子やスツールは、座る際に身体を支えるための家具です。"
            "脚部、座面、背もたれ、快適性に影響する素材などで構成されます。\n"
            "次に、あなたが選んだ対象について説明してください（姿勢、使用者、環境、主な構造）。"
        ),
        "ph_descriptors": (
            "Describe problems of the object you chose in under 3 sentences. "
            "Focus on dissatisfaction or pain points. "
            "Avoid praise or positive descriptions."
        ),
        "ph_descriptors_ja": (
            "選んだ対象の問題点を3文以内で記述してください。"
            "不満点や身体的負担などの課題に焦点を当ててください。"
            "良い点やポジティブな表現は避けてください。"
        ),
        "ph_descriptors_example": (
            "Uncomfortable after sitting for long periods.\n"
            "Hard to adjust posture.\n"
            "Takes up too much space when not in use."
        ),
        "ph_descriptors_example_ja": (
            "長時間座ると不快になる。\n"
            "姿勢を調整しにくい。\n"
            "使用していない時に場所を取りすぎる。"
        ),
        "ph_concept": (
            "Based on the problems you described, propose an idea or design to improve or solve them (3–6 sentences). "
            "If additives are shown, use at least ONE additive and explain what feature inspired you and how it relates to your idea. "
            "Your submission can be a related solution rather than the chair itself."
        ),
        "ph_concept_ja": (
            "記述した問題点をもとに、それらを改善または解決するためのアイデアやデザインを3〜6文で提案してください。"
            "Additiveが提示されている場合は、少なくとも1つを使用し、その特徴とアイデアの関係を説明してください。"
            "提出物は椅子そのものではなく、関連するサポート製品でも構いません。"
        ),
        "ph_concept_example": (
            "To solve comfort and space issues, I propose a seating support system that adapts to different postures throughout the day. "
            "Inspired by the additive feature of adaptive layering, the design uses flexible support layers that respond to body movement. "
            "Instead of a full chair, this concept could be a portable seat support compatible with existing furniture."
        ),
        "ph_concept_example_ja": (
            "快適性と省スペースの問題を解決するため、1日の中で変化する姿勢に対応する座面サポートシステムを提案します。"
            "Additiveの特徴であるアダプティブ・レイヤリングに着想を得て、身体の動きに反応する柔軟な支持層を採用します。"
            "フルサイズの椅子ではなく、既存の家具と併用できる携帯型サポートとして展開します。"
        ),
        "visual": "visuals/all_chair.png",
    },

    4: {
        "en_name": "Umbrella",
        "ja_name": "傘",
        "en_desc": (
            "Umbrella-type task: similar rain or sun protection products are acceptable "
            "(folding umbrella / parasol / raincoat / poncho, etc.). "
            "Choose ONE object you want to work with and generate an idea or design."
        ),
        "ja_desc": (
            "傘系課題：折りたたみ傘／日傘／レインコート／ポンチョなど、類似する雨・日よけ製品はすべて「傘」として扱い、提出OKです。"
            "ここで扱いたい対象を1つ選び、アイデア／デザインを考えてください。"
        ),
        "ph_base_desc": (
            "Example: An umbrella or parasol is a portable product used to protect from rain or sunlight. "
            "It usually has a frame, fabric, and a handle.\n"
            "Now describe the object YOU choose (weather context, carrying, and how it is used)."
        ),
        "ph_base_desc_ja": (
            "例：傘や日傘は、雨や日差しから身を守るための携帯用製品です。"
            "通常、骨組み、布地、持ち手で構成されています。\n"
            "次に、あなたが選んだ対象について説明してください（天候の状況、持ち運び方、使用方法）。"
        ),
        "ph_descriptors": (
            "Describe problems of the object you chose in under 3 sentences. "
            "Focus on dissatisfaction or pain points. "
            "Avoid praise or positive descriptions."
        ),
        "ph_descriptors_ja": (
            "選んだ対象の問題点を3文以内で記述してください。"
            "使いにくさや不満点などの課題に焦点を当ててください。"
            "良い点やポジティブな表現は避けてください。"
        ),
        "ph_descriptors_example": (
            "Difficult to dry after use.\n"
            "Drips water when stored.\n"
            "Unstable in strong wind."
        ),
        "ph_descriptors_example_ja": (
            "使用後に乾かしにくい。\n"
            "収納時に水滴が垂れる。\n"
            "強風時に不安定になる。"
        ),
        "ph_concept": (
            "Based on the problems you described, propose an idea or design to improve or solve them (3–6 sentences). "
            "If additives are shown, use at least ONE additive and explain what feature inspired you and how it relates to your idea. "
            "Your submission can be a related product or system, not necessarily an umbrella itself."
        ),
        "ph_concept_ja": (
            "記述した問題点をもとに、それらを改善または解決するためのアイデアやデザインを3〜6文で提案してください。"
            "Additiveが提示されている場合は、少なくとも1つを使用し、その特徴がどのようにアイデアに結びついているかを説明してください。"
            "提出物は傘そのものではなく、関連する製品やシステムでも構いません。"
        ),
        "ph_concept_example": (
            "To improve drying and storage problems, I propose a carry-and-dry solution integrated into everyday movement. "
            "Inspired by the additive feature of passive airflow channels, the design allows air circulation while the umbrella is stored. "
            "Rather than changing the umbrella itself, this idea could be a bag or holder that manages moisture and wind-related damage."
        ),
        "ph_concept_example_ja": (
            "乾燥や収納の問題を改善するため、日常動作の中で乾燥できるキャリー＆ドライの仕組みを提案します。"
            "Additiveの特徴である受動的な空気流路に着想を得て、収納中でも空気が循環する構造とします。"
            "傘本体を変更するのではなく、水分や風によるダメージを管理するバッグやホルダーとして展開します。"
        ),
        "visual": "visuals/all_umbrella.png",
    },
}

HINT_FUNC_BY_BASE = {
    1: get_pen_add,
    2: get_psu_add,
    3: get_chair_add,
    4: get_umbrella_add,
}

# =========================================================
# DB
# =========================================================

def _pick_working_db_path() -> str:
    """
    最小修正 + 更保險：
    - 優先用你設定的 DB_PATH（預設 instance/users.db）
    - 如果該路徑不可寫/不可建立，退回到 /tmp/users.db（Render 通常可寫）
    """
    p = DB_PATH

    try:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
        # 嘗試在該目錄建立/寫入測試檔（SQLite 要能在目錄寫入）
        test_file = os.path.join(d if d else ".", ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return p
    except Exception:
        return "/tmp/users.db"


_EFFECTIVE_DB_PATH = _pick_working_db_path()

def get_db_connection():
    conn = sqlite3.connect(_EFFECTIVE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema():
    conn = get_db_connection()
    cur = conn.cursor()

    # ✅ 1) 先確保 users 表存在（全新 DB 不會有）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        USERNAME TEXT UNIQUE,
        PASSWORD TEXT
    )
    """)
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute(
                "INSERT INTO users (USERNAME, PASSWORD) VALUES (?, ?)",
                ("admin", "admin")
            )
    for i in range(1, 31):
        username = f"usera{i:02d}"
        password = f"passa{i:02d}"
        cur.execute(
            "INSERT OR IGNORE INTO users (USERNAME, PASSWORD) VALUES (?, ?)",
            (username, password)
        )
    # ✅ 2) 再做欄位補齊（ALTER 之前一定要確保表存在）
    cur.execute("PRAGMA table_info(users)")
    cols = {row[1].upper() for row in cur.fetchall()}

    if "SEQUENCE" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN SEQUENCE TEXT")
    if "HINT_PATTERN" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN HINT_PATTERN TEXT")
    if "PROGRESS" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN PROGRESS INTEGER DEFAULT 0")

    # ✅ 3) 建 responses_v2（你原本就有）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS responses_v2 (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        USER_ID INTEGER NOT NULL,
        BASE_ID INTEGER NOT NULL,

        PHASE TEXT DEFAULT 'task',

        BASE_DESC TEXT,
        DESCRIPTORS TEXT,
        CONCEPT TEXT,

        IS_HINT INTEGER DEFAULT 0,
        HINT_INPUT_JSON TEXT,
        HINT_OUTPUT_JSON TEXT,

        TIME_SPENT_SEC INTEGER,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(USER_ID) REFERENCES users(id)
    )
    """)

    # Backward-compat: if old table exists and missing PHASE, add it.
    cur.execute("PRAGMA table_info(responses_v2)")
    rcols = {row[1].upper() for row in cur.fetchall()}
    if "PHASE" not in rcols:
        cur.execute("ALTER TABLE responses_v2 ADD COLUMN PHASE TEXT DEFAULT 'task'")

    conn.commit()
    conn.close()


# ✅ 最穩：第一次 request 才跑一次 schema（避免 gunicorn boot 時就死）
_schema_ready = False

@app.before_request
def init_schema_once():
    global _schema_ready
    if not _schema_ready:
        ensure_schema()
        _schema_ready = True
        # ✅ 如果 users 表是空的，自動建立第一個帳號
# =========================================================
# HELPERS
# =========================================================

def require_login():
    return "user_id" in session


def parse_sequence(text):
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip().isdigit()]


def parse_hint_pattern(text):
    mp = {}
    if not text:
        return mp
    for part in text.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        mp[int(k)] = v.strip() == "1"
    return mp


def hint_pattern_to_text(mp):
    return ",".join([f"{k}:{1 if mp[k] else 0}" for k in sorted(mp)])


def assign_conditions_if_needed(user_id: int):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT ID, SEQUENCE, HINT_PATTERN FROM users WHERE ID=?",
        (user_id,)
    ).fetchone()

    if user and user["SEQUENCE"] and user["HINT_PATTERN"]:
        conn.close()
        return

    if user_id % 2 == 0:
        seq = [1, 2, 3, 4]
        hint_map = {1: 1, 2: 0, 3: 1, 4: 0}
    else:
        seq = [3, 4, 1, 2]
        hint_map = {1: 0, 2: 1, 3: 0, 4: 1}

    conn.execute("""
        UPDATE users
        SET SEQUENCE=?, HINT_PATTERN=?, PROGRESS=0
        WHERE ID=?
    """, (
        ",".join(map(str, seq)),
        hint_pattern_to_text(hint_map),
        user_id
    ))

    conn.commit()
    conn.close()


def get_user_state(user_id: int):
    conn = get_db_connection()
    row = conn.execute("""
        SELECT ID, USERNAME, SEQUENCE, HINT_PATTERN, PROGRESS
        FROM users WHERE ID=?
    """, (user_id,)).fetchone()
    conn.close()

    seq = parse_sequence(row["SEQUENCE"])
    hint_map = parse_hint_pattern(row["HINT_PATTERN"])
    progress = int(row["PROGRESS"] or 0)
    return row, seq, hint_map, progress


def current_base_id(seq, progress):
    if progress >= len(seq):
        return 0
    return seq[progress]

def count_submissions(user_id: int) -> int:
    conn = get_db_connection()
    n = conn.execute(
        "SELECT COUNT(*) FROM responses_v2 WHERE USER_ID=?",
        (user_id,)
    ).fetchone()[0]
    conn.close()
    return int(n or 0)

# =========================================================
# SAVE RESPONSE
# =========================================================

def save_response_v2(user_id, base_id, is_hint):
    base_desc = request.form.get("base_desc")
    descriptors = request.form.get("descriptors")
    concept = request.form.get("concept")

    phase = request.form.get("phase") or "task"

    time_spent = request.form.get("time_spent_sec")
    try:
        time_spent = int(time_spent) if time_spent else None
    except:
        time_spent = None

    hint_input = request.form.get("hint_input_json")
    hint_output = request.form.get("hint_output_json")

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO responses_v2 (
            USER_ID, BASE_ID, IS_HINT,
            PHASE,
            BASE_DESC, DESCRIPTORS,
            CONCEPT,
            HINT_INPUT_JSON, HINT_OUTPUT_JSON,
            TIME_SPENT_SEC
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, base_id, int(is_hint),
        phase,
        base_desc, descriptors,
        concept,
        hint_input, hint_output,
        time_spent
    ))
    conn.commit()
    conn.close()

# =========================================================
# ROUTES
# =========================================================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        user = conn.execute("""
            SELECT * FROM users
            WHERE USERNAME=? AND PASSWORD=?
        """, (username, password)).fetchone()
        conn.close()

        if user:
            session.clear()
            session["user_id"] = user["ID"]
            assign_conditions_if_needed(user["ID"])
            return redirect(url_for("resume"))

        return "Invalid login", 401

    return render_template("login.html")


@app.route("/resume")
def resume():
    if not require_login():
        return redirect(url_for("login"))

    stage = session.get("stage")
    if stage is None:
        return redirect(url_for("instructions"))
    if stage == "instr1_done":
        return redirect(url_for("instructions2"))
    if stage == "instr2_done":
        return redirect(url_for("bases_intro"))

    user_id = session["user_id"]
    assign_conditions_if_needed(user_id)

    if count_submissions(user_id) >= 4:
        return redirect(url_for("finished"))

    _, seq, _, progress = get_user_state(user_id)
    nxt = current_base_id(seq, progress)

    if nxt == 0:
        return redirect(url_for("finished"))

    return redirect(url_for("base", base_id=nxt))

@app.route("/instructions", methods=["GET", "POST"])
def instructions():
    if not require_login():
        return redirect(url_for("login"))

    if request.method == "POST":
        session["stage"] = "instr1_done"
        session.modified = True
        return redirect(url_for("instructions2"))

    return render_template("instructions.html")


@app.route("/instructions2", methods=["GET", "POST"])
def instructions2():
    if not require_login():
        return redirect(url_for("login"))

    if session.get("stage") not in ("instr1_done", "bases_done"):
        return redirect(url_for("instructions"))

    if request.method == "POST":
        session["stage"] = "bases_done"
        session.modified = True
        return redirect(url_for("resume"))

    return render_template("instructions2.html")


@app.route("/base/<int:base_id>", methods=["GET", "POST"])
def base(base_id):
    if not require_login():
        return redirect(url_for("login"))

    if session.get("stage") != "bases_done":
        return redirect(url_for("resume"))

    user_id = session["user_id"]

    row, seq, hint_map, progress = get_user_state(user_id)
    expected = current_base_id(seq, progress)
    if base_id != expected:
        return redirect(url_for("base", base_id=expected))

    base_meta = BASES_META.get(base_id)
    if not base_meta:
        abort(404)

    show_hint = bool(hint_map.get(base_id, False))

    if request.method == "POST":
        save_response_v2(user_id, base_id, show_hint)

        conn = get_db_connection()
        conn.execute("UPDATE users SET PROGRESS = PROGRESS + 1 WHERE ID=?", (user_id,))
        conn.commit()
        conn.close()

        return redirect(url_for("resume"))

    return render_template(
        "base/bases.html",
        base_id=base_id,
        base_meta=base_meta,
        show_hint=show_hint,
        username=row["USERNAME"],
    )

@app.route("/finished")
def finished():
    if not require_login():
        return redirect(url_for("login"))

    user_id = session["user_id"]
    n = count_submissions(user_id)

    return render_template("finished.html", submitted_n=n, needed_n=4)

# =========================================================
# HINT API
# =========================================================

@app.route("/api/hint/<int:base_id>", methods=["POST"])
def api_hint(base_id):
    if not require_login():
        return {"error": "not_logged_in"}, 401

    fn = HINT_FUNC_BY_BASE.get(base_id)
    if fn is None:
        return {"error": "unknown_base_id"}, 404

    payload = request.get_json(silent=True) or {}

    descriptors = str(payload.get("descriptors", "")).strip()
    text = descriptors.strip()

    if not text:
        return {"additives": [], "debug": {}}

    try:
        out = fn(text, debug_json=True)
    except TypeError:
        out = fn(text)

    return {
        "additives": out.get("additives", []),
        "debug": out.get("debug", {}),
    }

# =========================================================
# ADMIN
# =========================================================

@app.route("/admin/export.csv")
def admin_export():
    pw = request.args.get("pw", "")
    if pw != ADMIN_PASSWORD:
        return "Forbidden", 403

    conn = get_db_connection()
    rows = conn.execute("""
        SELECT
            u.USERNAME,
            r.USER_ID,
            r.BASE_ID,
            r.IS_HINT,
            r.PHASE,
            r.BASE_DESC,
            r.DESCRIPTORS,
            r.CONCEPT,
            r.HINT_INPUT_JSON,
            r.HINT_OUTPUT_JSON,
            r.TIME_SPENT_SEC,
            r.CREATED_AT
        FROM responses_v2 r
        JOIN users u ON u.ID = r.USER_ID
        ORDER BY r.USER_ID, r.CREATED_AT
    """).fetchall()
    conn.close()

    output = StringIO()
    output.write("\ufeff")  # UTF-8 BOM

    writer = csv.writer(output)
    writer.writerow([
        "USERNAME", "USER_ID", "BASE_ID", "IS_HINT",
        "PHASE",
        "BASE_DESC", "DESCRIPTORS", "CONCEPT",
        "HINT_INPUT_JSON", "HINT_OUTPUT_JSON",
        "TIME_SPENT_SEC", "CREATED_AT"
    ])

    for r in rows:
        writer.writerow(list(r))

    resp = Response(output.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return resp


if __name__ == "__main__":
    app.run(debug=True, port=5001)