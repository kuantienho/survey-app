# scripts/run_pen_tool_v6_opposite_stable.py
# v6 (stable) + "opposite" additives mode
#
# Fixes added (minimal + stable):
# 1) Negative-signal extraction upgraded for pressure/burp/storage/mess cases
#    - Adds cues: pressure build-up, burp/burping, stored, spew/spit/ooze, mess
#    - Adds prototypes aligned to those mechanisms
# 2) Condition-bank "sanity gate" (NOT a pen-domain gate)
#    - Bank is treated as bonus NEG expansion, but we prevent obvious junk phrases
#    - Drops long/encyclopedic phrases, heavy-stopword phrases, digit-heavy phrases, etc.
#
# Bank policy:
# - Bank expands NEG only (bonus). We DO NOT require pen-domain keywords.
# - We DO require "sane phrase shape" (short, not encyclopedic).
#
# Output policy:
# - additives can be cross-domain (your "razor + chocolate" analogy).
# - final selection prefers "solution-ish" anchors while anti-matching NEG.
#
# CLI:
#   python scripts/run_pen_tool_v6_opposite_stable.py --text "..." [--debug-json]

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from paths import (
    PEN_OBJECTS_META,
    PEN_OBJECTS_FAISS,
    PEN_COND_BANK_FAISS,
    PEN_COND_BANK_META
)

# =========================
# Fixed policy knobs
# =========================
TARGET_TOPK = 9

CAND_K_DEFAULT = 260
EDGE_TOPK = 90
MMR_LAMBDA_DEFAULT = 0.75
MODEL_NAME_DEFAULT = "all-MiniLM-L6-v2"

BANK_TOPN_DEFAULT = 40
BANK_SEED_CAP_DEFAULT = 12
MIN_BANK_TO_USE_DEFAULT = 2

# Drift gate
DRIFT_AVG_THRESHOLD_DEFAULT = 0.36
DRIFT_ITEM_THRESHOLD_DEFAULT = 0.28

# v6: EDGE (phrase-shape) gate
# In opposite mode: we do NOT use vector sim gate in EDGE; only phrase-shape.
EDGE_SIM_MIN_DEFAULT = 0.23  # kept for compatibility/debug, but not used as hard gate in opposite mode

MIN_SIGNALS_DEFAULT = 2
MAX_PER_HEAD_DEFAULT = 2

ECHO_BAN_ENABLED_DEFAULT = True
SOFT_SUBSTITUTE_BAN_DEFAULT = False


# =========================
# Opposite policy knobs
# =========================
OPPOSITE_ENABLED_DEFAULT = True

# gates
NEG_SIM_Q_DEFAULT = 0.55   # keep candidates <= this quantile(sim_to_neg)
SOL_SIM_Q_DEFAULT = 0.55   # keep candidates >= this quantile(sim_to_sol) among survivors
SOL_SIM_MIN_DEFAULT = 0.0  # absolute floor (usually 0.0 is fine; use 0.15~0.25 if too random)

# scoring / query for MMR
A_SOL_DEFAULT = 1.0
B_NEG_DEFAULT = 1.0

# "solution-ish" anchors (mechanism/affordance oriented; cross-domain allowed)
# You can tune this list; it defines what "chocolate-role" means in your method.
_SOLUTION_ANCHORS = [
    "seal", "gasket", "o-ring", "valve", "vent", "pressure relief", "relief valve",
    "membrane", "barrier", "liner", "reservoir", "buffer", "absorber", "absorbent",
    "wick", "filter", "foam", "sponge", "capillary", "porous material",
    "clamp", "retainer", "stopper", "plug", "cover", "shield",
    "channel", "nozzle", "diffuser", "baffle",
    "lubricant", "grease", "wax",
    "scraper", "brush", "cleanup",
]

# =========================
# Paths (auto-detect)
# =========================
def _find_data_dir(start_dir: str) -> Tuple[str, str]:
    """
    Return (layout, data_dir)
    layout:
      - "website": <root>/data/pen
      - "proto":   <root>/data
    """
    cur = os.path.abspath(start_dir)
    for _ in range(7):
        cand_pen = os.path.join(cur, "data", "pen")
        cand_data = os.path.join(cur, "data")
        if os.path.isdir(cand_pen):
            return "website", cand_pen
        if os.path.isdir(cand_data):
            return "proto", cand_data
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return "fallback", os.path.join(os.path.abspath(start_dir), "data")

# =========================
# NLP / model
# =========================
_NLP = None
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP

def get_model(model_name: str = MODEL_NAME_DEFAULT):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# =========================
# Normalization / helpers
# =========================
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\"“”‘’`]", "", s)
    s = re.sub(r"[^a-z0-9\s\-\+\.']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canon_key(s: str) -> str:
    s0 = _norm(s)
    toks = s0.split()
    if len(toks) == 1:
        w = toks[0]
        if len(w) >= 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        return w
    return s0

def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x0 = _norm(x)
        if not x0:
            continue
        key = _canon_key(x0)
        if key in seen:
            continue
        seen.add(key)
        out.append(x0)
    return out


# =========================
# Small gate: number + object phrase ban
# =========================
_NUM_OBJECT_PHRASE = re.compile(r"(^|\s)\d{1,6}\s+[a-z][a-z0-9\-]{1,}\b", re.I)
_MM_OK = re.compile(r"^\d+(?:\.\d+)?\s*mm$", re.I)

def _has_number_object_phrase(s: str) -> bool:
    s0 = _norm(s)
    if not s0:
        return False
    toks = s0.split()
    if len(toks) < 2:
        return False
    if _MM_OK.match(s0.replace(" ", "")) or _MM_OK.match(s0):
        return False
    return bool(_NUM_OBJECT_PHRASE.search(s0))


# =========================
# Output junk / pronoun / human bans
# =========================
BAD_START_OUT = re.compile(
    r"^(the|a|an|this|that|these|those|it|its|they|them|their|he|she|his|her|my|our|your)\b",
    re.I
)
QUANT_PREFIX_OUT = re.compile(
    r"^(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"another|other|either|neither|each|every|any|some|many|few|several|"
    r"same|different|various|certain)\b",
    re.I
)

HUMAN_ROLE_NOUNS = {
    "user","users","operator","operators","consumer","consumers","customer","customers",
    "client","clients","technician","technicians","engineer","engineers","staff","crew",
    "people","person","persons","worker","workers","admin","administrator","administrators",
    "writer","writers","reader","readers","author","authors","editor","editors",
    "printer","printers","publisher","publishers","translator","translators",
    "illustrator","illustrators","scribe","scribes",
}

def _starts_with_bad_determiner(s: str) -> bool:
    return bool(BAD_START_OUT.search(_norm(s)))

def _starts_with_quant_prefix(s: str) -> bool:
    return bool(QUANT_PREFIX_OUT.search(_norm(s)))

def _contains_human_role(s: str) -> bool:
    toks = _norm(s).split()
    return any(t in HUMAN_ROLE_NOUNS for t in toks)


# =========================
# Pen component + soft substitute bans (output-side)
# =========================
PEN_COMPONENT_BAN_HARD = {
    "pen","pens","pencil","pencils","marker","markers","stylus","styli",
    "ink","nib","tip","feed","barrel","grip","cap","clip",
    "refill","cartridge","converter","reservoir","inkwell",
    "seal","o-ring","spring","button","mechanism","plunger",
    "paper",
    "write","writing","writer","scribble","draw","drawing",
}

PEN_SUBSTITUTE_BAN_SOFT = {
    "pencil","marker","highlighter","crayon","chalk","sharpie",
    "stylus","tablet","notebook","journal",
}

def _is_pen_component_or_core(s: str) -> bool:
    s0 = _norm(s)
    if not s0:
        return False
    toks = s0.split()
    if not toks:
        return False
    if len(toks) == 1 and toks[0] in PEN_COMPONENT_BAN_HARD:
        return True
    if any(t in PEN_COMPONENT_BAN_HARD for t in toks):
        return True
    return False

def _is_pen_substitute_soft(s: str) -> bool:
    toks = _norm(s).split()
    return any(t in PEN_SUBSTITUTE_BAN_SOFT for t in toks)


# =========================
# v6 two-layer TEXT DOMAIN ban (output-side)
# =========================
TEXT_DOMAIN_TOKEN_BAN = {
    "print","prints","printed","printing",
    "read","reads","reading","reader","readers",
    "write","writes","writing","writings","writer","writers",
    "publish","publishes","publishing","publisher","publishers",
    "edit","edits","editing","editor","editors",
    "book","books","page","pages","paragraph","paragraphs",
    "letter","letters","word","words","keyword","keywords",
}

def _is_text_domain_phrase_two_layer(s: str) -> bool:
    s0 = _norm(s)
    if not s0:
        return False
    toks = s0.split()
    if not toks:
        return False
    if len(toks) == 1 and toks[0] in TEXT_DOMAIN_TOKEN_BAN:
        return True
    if len(toks) <= 2 and toks[-1] in TEXT_DOMAIN_TOKEN_BAN:
        return True
    return False


# =========================
# Echo-ban
# =========================
def _input_content_lemmas(text: str) -> set:
    nlp = get_nlp()
    doc = nlp(text or "")
    out = set()
    for t in doc:
        if t.is_space or t.is_punct or t.is_stop:
            continue
        w = _norm(t.lemma_)
        if len(w) < 3:
            continue
        if w in {"not", "no", "too", "very"}:
            continue
        out.add(w)
    return out

def _echo_ban_hit(candidate: str, input_lemmas: set) -> bool:
    if not candidate or not input_lemmas:
        return False
    doc = get_nlp()(candidate)
    for t in doc:
        if t.is_space or t.is_punct or t.is_stop:
            continue
        w = _norm(t.lemma_)
        if len(w) < 3:
            continue
        if w in input_lemmas:
            return True
    return False


# =========================
# Negative signal extraction (rule + fallback)  [FIX #1]
# =========================
GENERIC_ADJ_STOP = {
    "new","old","other","same","different","first","last","many","few","most","some",
    "large","small","big","high","low","long","short"
}

# Upgraded cues: pressure build-up, burp, stored, spew/spit/ooze, mess
ISSUE_VERB_CUE = re.compile(
    r"\b("
    r"leak|leaks|leaking|leakage|"
    r"clog|clogs|clogged|clogging|"
    r"jam|jams|jammed|jamming|"
    r"skip|skips|skipping|"
    r"smudge|smudges|smudging|"
    r"bleed|bleeds|bleeding|"
    r"blot|blots|blotting|"
    r"dry\s*out|dries\s*out|drying\s*out|"
    r"scratch|scratches|scratching|scratchy|"
    r"fade|fades|fading|"
    r"flood|floods|flooding|"
    r"feather|feathers|feathering|"
    r"burp|burps|burping|"
    r"spew|spews|spewing|spit|spits|spitting|"
    r"ooze|oozes|oozing|"
    r"pool|pools|pooling|"
    r"creep|creeps|creeping|"
    r"mess|messy|"
    r"pressure|pressur(e|ized|izing|ization)|overpressur(e|ized)|"
    r"won['’]?t\s+write|doesn['’]?t\s+write|didn['’]?t\s+write|"
    r"stops?\s+writing|stopped\s+writing|stopping\s+writing|"
    r"ink\s+not\s+flowing|hard\s+start|inconsistent"
    r")\b",
    re.I
)

PHYS_PATTERNS = [
    re.compile(r"\btoo\s+(?P<x>[a-z][a-z\-]{1,30})\b", re.I),
    re.compile(r"\bnot\s+(?P<x>[a-z][a-z\-]{1,30})\b", re.I),
    re.compile(r"\b(no|poor|bad)\s+(?P<x>[a-z][a-z\-]{1,30})\b", re.I),

    re.compile(r"\b(ink\s+not\s+flowing|not\s+flowing)\b", re.I),
    re.compile(r"\b(hard\s+start|won['’]?t\s+start|doesn['’]?t\s+start)\b", re.I),

    # pressure / storage / burp patterns
    re.compile(r"\b(pressure\s+builds?\s+up|builds?\s+up\s+pressure|pressure\s+buildup)\b", re.I),
    re.compile(r"\b(stored\s+in\s+cap|stored\s+for\s+a\s+while|after\s+a\s+while)\b", re.I),
    re.compile(r"\b(burps?\s+suddenly|sudden\s+burp)\b", re.I),

    # mm allowance
    re.compile(r"\b(\d+(?:\.\d+)?\s*mm|0\.\d+\s*mm)\b", re.I),
]

# Updated prototypes: include pressure/burp/storage/mess
_COMPLAINT_PROTOTYPES = [
    "pen skips while writing",
    "ink dries out quickly",
    "pen won't write",
    "ink not flowing",
    "fountain pen hard start",
    "ink leaks into cap",
    "pressure builds up in cap",
    "pen burps suddenly",
    "ink spits and makes a mess",
    "scratchy nib",
    "cap doesn't seal",
    "smudges and bleeds through paper",
    "ink floods and feathers on paper",
    "inconsistent ink flow",
]

def extract_negative_signals_rule(text: str, max_signals: int = 25) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp(text)
    signals: List[str] = []

    # (1) adjectives
    for tok in doc:
        if tok.pos_ == "ADJ":
            w = tok.lemma_.lower()
            if w in GENERIC_ADJ_STOP:
                continue
            if len(w) < 3:
                continue
            signals.append(w)

    # (2) issue verbs / cues
    blob = _norm(text)
    for m in ISSUE_VERB_CUE.finditer(blob):
        signals.append(_norm(m.group(0)))

    # (3) noun chunks that look issue-ish (add pressure/storage/mess terms)
    issue_kw = (
        r"(ink|flow|nib|tip|feed|cartridge|converter|refill|cap|seal|o\-ring|paper|"
        r"leak|clog|jam|skip|smudge|bleed|blot|dry|scratch|fade|flood|feather|write|"
        r"burp|pressure|buildup|mess|spew|spit|ooze|pool|creep|store|stored)"
    )
    for chunk in doc.noun_chunks:
        t = _norm(chunk.text)
        if not t:
            continue
        if len(t.split()) > 8:
            continue
        has_adj = any(tk.pos_ == "ADJ" for tk in chunk)
        has_issue_word = bool(re.search(rf"\b{issue_kw}\b", t))
        if has_adj or has_issue_word:
            signals.append(t)

    # (4) regex patterns
    for pat in PHYS_PATTERNS:
        for m in pat.finditer(blob):
            seg = _norm(m.group(0))
            if seg and len(seg) >= 3:
                signals.append(seg)

    return _dedup_keep_order(signals)[:max_signals]

def extract_negative_signals_with_fallback(
    text: str,
    model: SentenceTransformer,
    max_signals: int = 25,
    min_signals: int = MIN_SIGNALS_DEFAULT,
) -> Tuple[List[str], Dict[str, Any]]:
    dbg = {"used_fallback": False, "rule_signals_n": 0, "fallback_top": []}
    rule = extract_negative_signals_rule(text, max_signals=max_signals)
    dbg["rule_signals_n"] = len(rule)
    if len(rule) >= min_signals:
        return rule, dbg

    desc = _norm(text)
    if not desc:
        return rule, dbg

    dbg["used_fallback"] = True
    protos = _COMPLAINT_PROTOTYPES[:]
    emb_p = np.asarray(model.encode(protos, normalize_embeddings=True), dtype="float32")
    emb_d = np.asarray(model.encode([desc], normalize_embeddings=True), dtype="float32")
    sims = (emb_p @ emb_d[0]).tolist()

    pairs = list(zip(protos, sims))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:5]
    dbg["fallback_top"] = [{"proto": p, "sim": float(s)} for p, s in top]

    add = [p for p, _ in top[:3]]
    merged = _dedup_keep_order(rule + add)
    return merged[:max_signals], dbg


# =========================
# MMR selection
# =========================
def mmr_select(
    emb: np.ndarray,
    q: np.ndarray,
    k: int = 60,
    lambda_: float = MMR_LAMBDA_DEFAULT,
) -> List[int]:
    sim_to_q = emb @ q
    n = emb.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    selected = []
    candidates = list(range(n))

    first = int(np.argmax(sim_to_q))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e9
        for i in candidates:
            rel = float(sim_to_q[i])
            div = max(float(emb[i] @ emb[j]) for j in selected)
            score = lambda_ * rel - (1 - lambda_) * div
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        candidates.remove(best_i)

    return selected


# =========================
# Bank sanity gate (NOT pen-domain gate)  [FIX #2]
# =========================
_BANK_MAX_TOKENS = 6
_BANK_STOPWORD_RATIO_MAX = 0.45

def _bank_phrase_sane(p: str) -> Tuple[bool, str]:
    """
    Keep bank phrases short & non-encyclopedic.
    This is NOT a pen-domain filter; it's just to prevent junk phrases
    like "most living and recently extinct dormice genera".
    """
    p0 = _norm(p)
    if not p0:
        return False, "empty"

    # too long => encyclopedic
    toks = p0.split()
    if len(toks) > _BANK_MAX_TOKENS:
        return False, "too_long"

    # digits => usually garbage from wiki
    if any(ch.isdigit() for ch in p0):
        # allow mm-only if you really want; but bank should rarely contain mm
        if not _MM_OK.match(p0.replace(" ", "")) and not _MM_OK.match(p0):
            return False, "has_digits"

    # common encyclopedic patterns
    if " and " in p0 and len(toks) >= 5:
        return False, "encyclopedic_and"
    if " of " in p0 and len(toks) >= 5:
        return False, "encyclopedic_of"

    doc = get_nlp()(p0)
    if not doc:
        return False, "nlp_fail"

    if len(doc) >= 3:
        sw = sum(1 for t in doc if t.is_stop)
        if (sw / len(doc)) > _BANK_STOPWORD_RATIO_MAX:
            return False, "stopword_heavy"

    # reject phrases that look like whole clauses (verbs)
    if any(t.pos_ in ("VERB", "AUX") for t in doc):
        return False, "has_verb"

    return True, ""


# =========================
# Condition bank expansion + drift
# =========================
@dataclass
class PenResources:
    obj_texts: List[str]
    obj_index: faiss.Index
    bank_phrases: List[str]
    bank_index: faiss.Index
    bank_name: str
    model_name: str = MODEL_NAME_DEFAULT

_RES: Optional[PenResources] = None

def _load_bank(bank_csv: str, bank_faiss: str, model_name: str):
    df = pd.read_csv(bank_csv)

    if "phrase_norm" in df.columns:
        phrases = df["phrase_norm"].astype(str).tolist()
    elif "phrase" in df.columns:
        phrases = df["phrase"].astype(str).tolist()
    else:
        phrases = df.iloc[:, 0].astype(str).tolist()

    idx = faiss.read_index(bank_faiss)
    return phrases, idx

    # (如果沒有 faiss 檔，你原本應該在這裡建 index；略)

    model = get_model(model_name)
    emb = np.asarray(model.encode(phrases, normalize_embeddings=True, show_progress_bar=True), dtype="float32")
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return phrases, idx

def load_pen_resources(model_name: str = MODEL_NAME_DEFAULT) -> PenResources:
    global _RES
    if _RES is not None:
        return _RES

    if not os.path.exists(PEN_OBJECTS_META):
        raise FileNotFoundError(f"Missing: {PEN_OBJECTS_META}")
    if not os.path.exists(PEN_OBJECTS_FAISS):
        raise FileNotFoundError(f"Missing: {PEN_OBJECTS_FAISS}")

    obj_df = pd.read_csv(PEN_OBJECTS_META)
    if "object" in obj_df.columns:
        obj_texts = obj_df["object"].astype(str).tolist()
    elif "term" in obj_df.columns:
        obj_texts = obj_df["term"].astype(str).tolist()
    else:
        raise ValueError("pen_objects_meta.csv must contain column 'object' or 'term'")
    obj_index = faiss.read_index(PEN_OBJECTS_FAISS)

    # use indexed condition bank (meta + faiss)
    if not os.path.exists(PEN_COND_BANK_META):
        raise FileNotFoundError(f"Missing: {PEN_COND_BANK_META}")
    if not os.path.exists(PEN_COND_BANK_FAISS):
        raise FileNotFoundError(f"Missing: {PEN_COND_BANK_FAISS}")

    bank_phrases, bank_index = _load_bank(
        PEN_COND_BANK_META,
        PEN_COND_BANK_FAISS,
        model_name
    )
    bank_name = "indexed"

    _RES = PenResources(
        obj_texts=obj_texts,
        obj_index=obj_index,
        bank_phrases=bank_phrases,
        bank_index=bank_index,
        bank_name=bank_name,
        model_name=model_name,
    )
    return _RES

def expand_conditions_with_bank_bonus(
    signals: List[str],
    res: PenResources,
    topn: int = BANK_TOPN_DEFAULT,
) -> Tuple[List[str], List[Dict[str, float]], Dict[str, Any]]:
    """
    Bank expands NEG phrases only (bonus).
    NO pen-domain gate here.
    YES sanity gate here.
    """
    gate_dbg = {
        "bank_available": True,
        "bank_name": res.bank_name,
        "bank_csv": PEN_COND_BANK_META,
        "bank_faiss": PEN_COND_BANK_FAISS,
        "retrieved_n": 0,
        "returned_n": 0,
        "dropped_dup_or_empty": 0,
        "dropped_sanity": 0,
        "dropped_sanity_reasons": {},
    }

    if not signals:
        return [], [], gate_dbg

    model = get_model(res.model_name)
    sig_emb = np.asarray(model.encode(signals, normalize_embeddings=True), dtype="float32")
    q = sig_emb.mean(axis=0)
    q = q / (np.linalg.norm(q) + 1e-12)
    q = np.asarray([q], dtype="float32")

    D, I = res.bank_index.search(q, topn)
    idxs = I[0].tolist()
    sims = D[0].tolist()
    gate_dbg["retrieved_n"] = len([i for i in idxs if i is not None])

    out = []
    seen = set()
    dbg = []
    for i, sc in zip(idxs, sims):
        if i < 0 or i >= len(res.bank_phrases):
            continue
        p = _norm(res.bank_phrases[i])
        if not p:
            gate_dbg["dropped_dup_or_empty"] += 1
            continue
        if p in seen:
            gate_dbg["dropped_dup_or_empty"] += 1
            continue

        ok, reason = _bank_phrase_sane(p)
        if not ok:
            gate_dbg["dropped_sanity"] += 1
            gate_dbg["dropped_sanity_reasons"][reason] = gate_dbg["dropped_sanity_reasons"].get(reason, 0) + 1
            continue

        seen.add(p)
        out.append(p)
        if len(dbg) < 20:
            dbg.append({"phrase": p, "score": float(sc)})

    gate_dbg["returned_n"] = len(out)
    return out, dbg, gate_dbg

def bank_drift_filter_expanded(
    signals: List[str],
    expanded: List[str],
    model: SentenceTransformer,
    keep_top: int = BANK_TOPN_DEFAULT,
    avg_sim_threshold: float = DRIFT_AVG_THRESHOLD_DEFAULT,
    per_item_threshold: float = DRIFT_ITEM_THRESHOLD_DEFAULT,
) -> Tuple[List[str], Dict[str, Any]]:
    dbg = {
        "enabled": True,
        "avg_sim_threshold": float(avg_sim_threshold),
        "per_item_threshold": float(per_item_threshold),
        "signals_n": len(signals),
        "expanded_n": len(expanded),
        "kept_n": 0,
        "avg_sim_kept": None,
        "drifted": False,
    }

    sig = [_norm(s) for s in signals if _norm(s)]
    exp = [_norm(x) for x in expanded if _norm(x)]
    if not sig or not exp:
        dbg["kept_n"] = 0
        dbg["drifted"] = True if exp and not sig else False
        return [], dbg

    sig_emb = np.asarray(model.encode(sig, normalize_embeddings=True), dtype="float32")
    centroid = sig_emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    exp_emb = np.asarray(model.encode(exp, normalize_embeddings=True), dtype="float32")
    sims = (exp_emb @ centroid).tolist()

    pairs = list(zip(exp, sims))
    kept_pairs = [(p, s) for p, s in pairs if s >= per_item_threshold]
    kept_pairs.sort(key=lambda x: x[1], reverse=True)

    kept = [p for p, _ in kept_pairs[:keep_top]]
    dbg["kept_n"] = len(kept)

    if kept_pairs:
        avg_sim = float(np.mean([s for _, s in kept_pairs[:min(len(kept_pairs), 20)]]))
    else:
        avg_sim = 0.0
    dbg["avg_sim_kept"] = avg_sim

    if avg_sim < avg_sim_threshold:
        dbg["drifted"] = True
        return [], dbg

    return kept, dbg


# =========================
# EDGE phrase-shape filter (strict-ish)
# =========================
_EDGE_ICS_DROP = re.compile(r"^[a-z]{4,}ics$", re.I)
_EDGE_MAX_TOKENS = 3

EDGE_UNIGRAM_BAN = {
    "thing","things","stuff","object","objects","item","items",
    "word","words","keyword","keywords","letter","letters",
    "usage","phenomena","product","products","general","semantic","semantics",
    "problem","issue","issues","quality","performance","reliability","stability",
    "system","method","type","kind","concept","idea","information","data","detail","details",
}

EDGE_ABSTRACT_SUFFIX = (
    "tion","sion","ment","ness","ity","ism","ance","ence","ship","acy","dom","hood"
)
EDGE_ABSTRACT_WHITELIST = {"station","attachment","segment","instrument","ornament","basement","garment","fragment"}

def _is_edge_phrase(s: str) -> bool:
    s0 = _norm(s)
    if not s0:
        return False

    if _starts_with_bad_determiner(s0) or _starts_with_quant_prefix(s0):
        return False
    if _contains_human_role(s0):
        return False

    if s0 in EDGE_UNIGRAM_BAN:
        return False
    if _EDGE_ICS_DROP.match(s0):
        return False

    toks = s0.split()
    if len(toks) > _EDGE_MAX_TOKENS:
        return False

    if any(ch.isdigit() for ch in s0):
        if re.fullmatch(r"\d+(?:\.\d+)?\s*mm", s0) or re.fullmatch(r"\d+(?:\.\d+)?mm", s0):
            pass
        else:
            return False

    if re.search(r"(^-|-$)", s0):
        return False

    doc = get_nlp()(s0)
    if not doc:
        return False

    if len(doc) >= 2:
        sw = sum(1 for t in doc if t.is_stop)
        if (sw / len(doc)) > 0.30:
            return False

    # no verbs
    if any(t.pos_ in ("VERB", "AUX") for t in doc):
        return False

    pos = [t.pos_ for t in doc]

    allowed = False
    if len(doc) == 1 and pos[0] in ("NOUN", "PROPN"):
        allowed = True
    elif pos == ["ADJ", "NOUN"]:
        allowed = True
    elif pos == ["NOUN", "NOUN"]:
        allowed = True
    elif len(pos) == 3 and pos[0] == "ADJ" and pos[1] == "ADJ" and pos[2] == "NOUN":
        allowed = True
    elif len(pos) == 3 and pos[0] == "ADJ" and pos[1] == "NOUN" and pos[2] == "NOUN":
        allowed = True

    if not allowed:
        return False

    # abstract suffix drop on last token (unless whitelisted)
    last_tok = doc[-1]
    last = _norm(last_tok.text)
    if last_tok.pos_ != "PROPN":
        if last not in EDGE_ABSTRACT_WHITELIST and len(last) >= 6:
            for suf in EDGE_ABSTRACT_SUFFIX:
                if last.endswith(suf):
                    return False

    return True

def _topk_pool_edge_shape_only(
    pool_ranked: List[str],
    k: int,
) -> Tuple[List[str], Dict[str, Any]]:
    dbg = {"dropped": {"not_edge_phrase": 0, "number_object_phrase": 0}}
    out = []
    if not pool_ranked:
        return [], dbg

    for x in pool_ranked:
        x0 = _norm(x)
        if _has_number_object_phrase(x0):
            dbg["dropped"]["number_object_phrase"] += 1
            continue
        if not _is_edge_phrase(x0):
            dbg["dropped"]["not_edge_phrase"] += 1
            continue
        out.append(x0)

    return _dedup_keep_order(out)[:k], dbg


# =========================
# Final selection: top-k closest with diversity caps
# =========================
def _score_to_query(cands: List[str], q: np.ndarray, model: SentenceTransformer) -> List[Tuple[str, float]]:
    if not cands:
        return []
    emb = np.asarray(model.encode(cands, normalize_embeddings=True), dtype="float32")
    q0 = np.asarray(q, dtype="float32").reshape(-1)
    sims = (emb @ q0).tolist()
    pairs = list(zip(cands, [float(s) for s in sims]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def _head_word_simple(s: str) -> str:
    toks = _norm(s).split()
    return toks[-1] if toks else ""

def compose_topk_close(
    edge_candidates: List[str],
    q: np.ndarray,
    model: SentenceTransformer,
    target_topk: int = TARGET_TOPK,
    max_per_head: int = MAX_PER_HEAD_DEFAULT,
) -> List[str]:
    if not edge_candidates:
        return []
    scored = _score_to_query(edge_candidates, q, model)
    used = set()
    head_count: Dict[str, int] = {}
    out: List[str] = []

    for x, _ in scored:
        if len(out) >= target_topk:
            break
        x0 = _norm(x)
        if not x0:
            continue
        key = _canon_key(x0)
        if key in used:
            continue
        head = _head_word_simple(x0)
        if head and head_count.get(head, 0) >= max_per_head:
            continue
        out.append(x0)
        used.add(key)
        if head:
            head_count[head] = head_count.get(head, 0) + 1

    return out[:target_topk]


# =========================
# Optional tiny quality ban (kept small)
# =========================
_QUALITY_EXACT_BAN = {
    "unknown", "none", "misc", "various", "different", "others"
}
_QUALITY_BAD_CHARS = re.compile(r"[^\x00-\x7F]")  # non-ascii -> often noisy dumps
_QUALITY_TRUNC_RE = re.compile(r".+\.\.\.$")
_QUALITY_BIOMOL_SUFFIX = ("ase", "in", "ine", "ate", "ose", "gen", "genic", "genesis")

def quality_ban(s: str) -> Tuple[bool, str]:
    s0 = _norm(s)
    if not s0:
        return True, "quality_empty"
    if s0 in _QUALITY_EXACT_BAN:
        return True, "quality_exact"
    if _QUALITY_BAD_CHARS.search(s):
        return True, "quality_bad_chars"
    if _QUALITY_TRUNC_RE.match(s0):
        return True, "quality_truncated"

    # very light biomolecule-ish suffix ban (kept tiny; avoid endless work)
    toks = s0.split()
    if len(toks) == 1:
        w = toks[0]
        if len(w) >= 8 and any(w.endswith(suf) for suf in _QUALITY_BIOMOL_SUFFIX):
            return True, "quality_biomolecule_suffix"

    return False, ""


# =========================
# Candidate bans (output-side)
# =========================
def _ban_candidate(
    cand: str,
    input_lemmas: set,
    enable_echo_ban: bool = True,
    enable_soft_substitute_ban: bool = False,
    enable_quality_ban: bool = True,
) -> Tuple[bool, str]:
    c0 = _norm(cand)
    if not c0:
        return True, "empty"

    if enable_quality_ban:
        qb, qreason = quality_ban(cand)
        if qb:
            return True, qreason

    if _starts_with_bad_determiner(c0) or _starts_with_quant_prefix(c0):
        return True, "bad_start"

    if _contains_human_role(c0):
        return True, "human_role"

    if _is_pen_component_or_core(c0):
        return True, "pen_component"

    if enable_soft_substitute_ban and _is_pen_substitute_soft(c0):
        return True, "pen_substitute"

    if _is_text_domain_phrase_two_layer(c0):
        return True, "text_domain"

    if enable_echo_ban and _echo_ban_hit(c0, input_lemmas):
        return True, "echo"

    return False, ""


# =========================
# Opposite gate: anti-neg + solution-ish gate
# =========================
def _safe_norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32").reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)

def build_query_from_phrases(phrases: List[str], model: SentenceTransformer) -> Optional[np.ndarray]:
    ps = [_norm(p) for p in (phrases or []) if _norm(p)]
    if not ps:
        return None
    emb = np.asarray(model.encode(ps, normalize_embeddings=True), dtype="float32")
    q = emb.mean(axis=0)
    return _safe_norm_vec(q)

def opposite_rank_pool(
    pool: List[str],
    model: SentenceTransformer,
    neg_phrases: List[str],
    solution_anchors: List[str],
    a_sol: float = A_SOL_DEFAULT,
    b_neg: float = B_NEG_DEFAULT,
    neg_sim_q: float = NEG_SIM_Q_DEFAULT,
    sol_sim_q: float = SOL_SIM_Q_DEFAULT,
    sol_sim_min: float = SOL_SIM_MIN_DEFAULT,
) -> Tuple[List[str], Dict[str, Any], np.ndarray]:
    """
    Returns:
      ranked_pool: list[str]
      dbg: debug dict
      q_rank: vector used for MMR relevance
    """
    dbg = {
        "enabled": True,
        "a_sol": float(a_sol),
        "b_neg": float(b_neg),
        "neg_sim_q": float(neg_sim_q),
        "sol_sim_q": float(sol_sim_q),
        "sol_sim_min": float(sol_sim_min),
        "pool_n": int(len(pool)),
        "kept_after_anti_neg_gate": 0,
        "kept_after_solution_gate": 0,
        "neg_sim_max_used": None,
        "sol_sim_min_used": None,
        "sim_neg_minmax": None,
        "sim_sol_minmax": None,
        "score_minmax": None,
        "notes": [],
    }

    if not pool:
        q_rank = np.zeros((384,), dtype="float32")  # unused
        return [], dbg, q_rank

    q_neg = build_query_from_phrases(neg_phrases, model)
    q_sol = build_query_from_phrases(solution_anchors, model)

    if q_neg is None or q_sol is None:
        dbg["notes"].append("q_neg or q_sol is None; skip opposite ranking")
        q_rank = q_sol if q_sol is not None else (q_neg if q_neg is not None else np.zeros((384,), dtype="float32"))
        return pool, dbg, q_rank

    q_rank = _safe_norm_vec(float(a_sol) * q_sol - float(b_neg) * q_neg)

    emb_pool = np.asarray(model.encode(pool, normalize_embeddings=True), dtype="float32")
    sim_neg = (emb_pool @ np.asarray(q_neg, dtype="float32")).reshape(-1)
    sim_sol = (emb_pool @ np.asarray(q_sol, dtype="float32")).reshape(-1)

    dbg["sim_neg_minmax"] = [float(np.min(sim_neg)), float(np.max(sim_neg))]
    dbg["sim_sol_minmax"] = [float(np.min(sim_sol)), float(np.max(sim_sol))]

    # anti-neg gate by quantile
    neg_thr = float(np.quantile(sim_neg, float(neg_sim_q)))
    keep1 = sim_neg <= neg_thr
    dbg["neg_sim_max_used"] = float(neg_thr)

    idx1 = np.where(keep1)[0].tolist()
    dbg["kept_after_anti_neg_gate"] = int(len(idx1))
    if not idx1:
        dbg["notes"].append("anti-neg gate removed all; fallback=keep all")
        idx1 = list(range(len(pool)))

    # solution gate among survivors by quantile + floor
    sim_sol_surv = sim_sol[idx1]
    sol_thr_q = float(np.quantile(sim_sol_surv, float(sol_sim_q))) if len(sim_sol_surv) else 0.0
    sol_thr = max(float(sol_thr_q), float(sol_sim_min))
    dbg["sol_sim_min_used"] = float(sol_thr)

    idx2 = [idx1[j] for j in range(len(idx1)) if float(sim_sol[idx1[j]]) >= sol_thr]
    dbg["kept_after_solution_gate"] = int(len(idx2))

    if not idx2:
        dbg["notes"].append("solution gate removed all; fallback=use anti-neg survivors")
        idx2 = idx1

    # score and rank
    score = float(a_sol) * sim_sol - float(b_neg) * sim_neg
    score2 = score[idx2]
    order = np.argsort(-score2)

    ranked = [pool[idx2[i]] for i in order.tolist()]

    dbg["score_minmax"] = [float(np.min(score2)), float(np.max(score2))] if len(score2) else None

    return ranked, dbg, q_rank


# =========================
# Main API
# =========================
def get_pen_add(
    text: str,
    debug_json: bool = False,
    opposite_enabled: bool = OPPOSITE_ENABLED_DEFAULT,
) -> Dict[str, Any]:
    res = load_pen_resources(model_name=MODEL_NAME_DEFAULT)
    model = get_model(res.model_name)

    input_lemmas = _input_content_lemmas(text)

    # 1) NEG signals
    signals, sig_dbg = extract_negative_signals_with_fallback(
        text=text, model=model, max_signals=25, min_signals=MIN_SIGNALS_DEFAULT
    )
    neg_phrases = signals[:]

    # 2) bank bonus NEG expansion (sanity-gated)
    expanded, bank_hits_dbg, bank_gate_dbg = expand_conditions_with_bank_bonus(
        signals=signals, res=res, topn=BANK_TOPN_DEFAULT
    )

    expanded_kept, drift_dbg = bank_drift_filter_expanded(
        signals=signals,
        expanded=expanded,
        model=model,
        keep_top=BANK_TOPN_DEFAULT,
        avg_sim_threshold=DRIFT_AVG_THRESHOLD_DEFAULT,
        per_item_threshold=DRIFT_ITEM_THRESHOLD_DEFAULT,
    )

    bank_bonus_used = (not drift_dbg.get("drifted", False)) and (len(expanded_kept) >= int(MIN_BANK_TO_USE_DEFAULT))
    if bank_bonus_used:
        neg_phrases = _dedup_keep_order(neg_phrases + expanded_kept[: int(BANK_SEED_CAP_DEFAULT)])

    # early exit if no neg phrases
    if not neg_phrases:
        out = {"additives": []}
        if debug_json:
            out["debug"] = {
                "signals": signals,
                "signals_debug": sig_dbg,
                "neg_phrases": neg_phrases,
                "expanded_top": expanded[:30],
                "expanded_kept": expanded_kept[:30],
                "bank_bonus_used": bank_bonus_used,
                "bank_gate": bank_gate_dbg,
                "bank_drift": drift_dbg,
                "bank_name": res.bank_name,
                "edge_dbg": {"dropped": {"not_edge_phrase": 0, "number_object_phrase": 0}},
                "ban_dropped": {},
                "counts": {"pool_candidates": 0, "pool_dedup": 0, "ranked_after_opposite": 0, "edge_take": 0, "filtered_after_ban": 0, "final": 0},
                "paths_resolved": {
                    "objects_meta": PEN_OBJECTS_META,
                    "objects_faiss": PEN_OBJECTS_FAISS,
                    "cond_bank_csv": PEN_COND_BANK_META,
                    "cond_bank_faiss": PEN_COND_BANK_FAISS,
                },
            }
        return out

    # 3) retrieve object pool using NEG query (stable baseline)
    q_neg = build_query_from_phrases(neg_phrases, model)
    if q_neg is None:
        out = {"additives": []}
        if debug_json:
            out["debug"] = {"signals": signals, "signals_debug": sig_dbg, "neg_phrases": neg_phrases, "notes": ["q_neg is None"]}
        return out

    D, I = res.obj_index.search(np.asarray([q_neg], dtype="float32"), CAND_K_DEFAULT)
    cand_idx = [i for i in I[0].tolist() if 0 <= i < len(res.obj_texts)]
    pool = [res.obj_texts[i] for i in cand_idx if res.obj_texts[i]]
    pool = _dedup_keep_order(pool)

    # 4) opposite gate (rank + build q_rank for MMR)
    opposite_dbg = {"enabled": bool(opposite_enabled), "used": False}
    ranked = pool[:]
    q_rank = q_neg

    if opposite_enabled:
        ranked2, rank_dbg, q_rank2 = opposite_rank_pool(
            pool=pool,
            model=model,
            neg_phrases=neg_phrases,
            solution_anchors=_SOLUTION_ANCHORS,
            a_sol=A_SOL_DEFAULT,
            b_neg=B_NEG_DEFAULT,
            neg_sim_q=NEG_SIM_Q_DEFAULT,
            sol_sim_q=SOL_SIM_Q_DEFAULT,
            sol_sim_min=SOL_SIM_MIN_DEFAULT,
        )
        ranked = ranked2
        q_rank = q_rank2
        opposite_dbg = {
            "enabled": True,
            "used": True,
            "a_sol": A_SOL_DEFAULT,
            "b_neg": B_NEG_DEFAULT,
            "q_rank": "normalize(a*q_solution - b*q_negative)",
            "rank_debug": rank_dbg,
            "anchors_n": len(_SOLUTION_ANCHORS),
        }

    # 5) MMR rerank on ranked candidates (use q_rank for relevance)
    emb_ranked = np.asarray(model.encode(ranked, normalize_embeddings=True), dtype="float32")
    sel = mmr_select(
        emb=emb_ranked,
        q=np.asarray(q_rank, dtype="float32"),
        k=min(len(ranked), max(EDGE_TOPK * 4, 120)),
        lambda_=MMR_LAMBDA_DEFAULT,
    )
    ranked_mmr = [ranked[i] for i in sel]

    # 6) edge phrase-shape only
    edge_take, edge_dbg = _topk_pool_edge_shape_only(
        ranked_mmr,
        k=min(EDGE_TOPK, max(EDGE_TOPK, 80)),
    )

    # 7) bans (output-side)
    ban_dropped: Dict[str, int] = {
        "pen_component": 0,
        "pen_substitute": 0,
        "echo": 0,
        "bad_start": 0,
        "human_role": 0,
        "empty": 0,
        "text_domain": 0,
        "quality_empty": 0,
        "quality_bad_chars": 0,
        "quality_exact": 0,
        "quality_truncated": 0,
        "quality_biomolecule_suffix": 0,
    }

    filtered = []
    for x in edge_take:
        banned, reason = _ban_candidate(
            x,
            input_lemmas=input_lemmas,
            enable_echo_ban=ECHO_BAN_ENABLED_DEFAULT,
            enable_soft_substitute_ban=SOFT_SUBSTITUTE_BAN_DEFAULT,
            enable_quality_ban=True,
        )
        if banned:
            ban_dropped[reason] = ban_dropped.get(reason, 0) + 1
            continue
        filtered.append(_norm(x))
    filtered = _dedup_keep_order(filtered)

    # 8) final selection
    final = compose_topk_close(
        edge_candidates=filtered,
        q=q_rank,
        model=model,
        target_topk=TARGET_TOPK,
        max_per_head=MAX_PER_HEAD_DEFAULT,
    )

    out = {"additives": final[:TARGET_TOPK]}
    if debug_json:
        out["debug"] = {
            "signals": signals,
            "signals_debug": sig_dbg,
            "neg_phrases": neg_phrases,
            "expanded_top": expanded[:30],
            "expanded_kept": expanded_kept[:30],
            "bank_bonus_used": bank_bonus_used,
            "bank_gate": bank_gate_dbg,
            "bank_drift": drift_dbg,
            "bank_name": res.bank_name,
            "opposite_gate": opposite_dbg,
            "mmr_input_n": int(len(ranked)),
            "edge_dbg": edge_dbg,
            "ban_dropped": ban_dropped,
            "counts": {
                "pool_candidates": int(len(cand_idx)),
                "pool_dedup": int(len(pool)),
                "ranked_after_opposite": int(len(ranked)),
                "edge_take": int(len(edge_take)),
                "filtered_after_ban": int(len(filtered)),
                "final": int(len(final)),
            },
            "paths_resolved": {
                "objects_meta": PEN_OBJECTS_META,
                "objects_faiss": PEN_OBJECTS_FAISS,
                "cond_bank_csv": PEN_COND_BANK_META,
                "cond_bank_faiss": PEN_COND_BANK_FAISS,
            },
            "policy": {
                "target_topk": TARGET_TOPK,
                "cand_k": CAND_K_DEFAULT,
                "edge_topk": EDGE_TOPK,
                "mmr_lambda": MMR_LAMBDA_DEFAULT,
                "echo_ban": ECHO_BAN_ENABLED_DEFAULT,
                "soft_substitute_ban": SOFT_SUBSTITUTE_BAN_DEFAULT,
                "max_per_head": MAX_PER_HEAD_DEFAULT,
                "bank_policy": "bank expands NEG only (bonus); NO pen-domain gate; YES sanity gate; used if not drifted and kept>=min_bank_to_use; cap=bank_seed_cap",
                "bank_sanity": {
                    "max_tokens": _BANK_MAX_TOKENS,
                    "stopword_ratio_max": _BANK_STOPWORD_RATIO_MAX,
                    "reject_verbs": True,
                },
                "opposite_policy": {
                    "enabled": opposite_enabled,
                    "anchors": "small mechanism/affordance list (cross-domain allowed)",
                    "neg_gate": f"sim_neg <= quantile(sim_neg, {NEG_SIM_Q_DEFAULT})",
                    "sol_gate": f"sim_sol >= max(quantile(sim_sol_on_survivors, {SOL_SIM_Q_DEFAULT}), {SOL_SIM_MIN_DEFAULT})",
                    "score": f"{A_SOL_DEFAULT}*sim_sol - {B_NEG_DEFAULT}*sim_neg",
                    "mmr_query": "normalize(a*q_sol - b*q_neg)",
                    "edge_filter": "shape-only",
                    "quality_ban": "enabled (tiny)",
                },
                "signal_fix": "added pressure/burp/storage/mess cues + prototypes",
            }
        }

    return out


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Describe pen issues (English).")
    ap.add_argument("--debug-json", action="store_true", help="Include debug section in output.")
    ap.add_argument("--no-opposite", action="store_true", help="Disable opposite gate (for debugging).")
    args = ap.parse_args()

    out = get_pen_add(
        text=args.text,
        debug_json=args.debug_json,
        opposite_enabled=(not args.no_opposite),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()