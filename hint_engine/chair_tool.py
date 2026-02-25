# scripts/run_chair_tool_v6_opposite_stable.py
# v6 (stable) + "opposite" additives mode  (CHAIR)
#
# Matches pen-tool v6 opposite stable LOGIC (domain-adapted for chair):
#
# Pipeline (same as pen v6 opposite):
#   NEG signals (rule + fallback)
#   -> condition bank bonus NEG expansion (WITH sanity gate) + drift filter
#   -> FAISS object pool retrieved by NEG query
#   -> opposite gate: anti-neg + solution-ish gate, builds q_rank = normalize(a*q_sol - b*q_neg)
#   -> MMR rerank using q_rank
#   -> EDGE gate (shape-only in opposite mode)
#   -> output-side bans (shared) + chair-specific bans
#   -> final Top-9 closest to q_rank with light diversity caps
#
# Bank policy:
# - Bank expands NEG only (bonus). No chair-domain requirement.
# - Bank must pass sanity gate (short, non-encyclopedic).
#
# Output policy:
# - additives can be cross-domain.
# - final selection prefers "solution-ish" anchors while anti-matching NEG.
#
# CLI:
#   python scripts/run_chair_tool_v6_opposite_stable.py --text "..." [--debug-json] [--no-opposite]

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
    CHAIR_OBJECTS_META,
    CHAIR_OBJECTS_FAISS,
    CHAIR_COND_BANK_META,
    CHAIR_COND_BANK_FAISS,
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

# Drift gate (chair tuned but same logic)
DRIFT_AVG_THRESHOLD_DEFAULT = 0.34
DRIFT_ITEM_THRESHOLD_DEFAULT = 0.26

MIN_SIGNALS_DEFAULT = 2
MAX_PER_HEAD_DEFAULT = 2

ECHO_BAN_ENABLED_DEFAULT = True


# =========================
# Opposite policy knobs (same logic as pen)
# =========================
OPPOSITE_ENABLED_DEFAULT = True

NEG_SIM_Q_DEFAULT = 0.55   # keep candidates <= this quantile(sim_to_neg)
SOL_SIM_Q_DEFAULT = 0.55   # keep candidates >= this quantile(sim_to_sol) among survivors
SOL_SIM_MIN_DEFAULT = 0.0  # floor

A_SOL_DEFAULT = 1.0
B_NEG_DEFAULT = 1.0

# "solution-ish" anchors for chair repair / stability / damping / anti-slip (cross-domain allowed)
_SOLUTION_ANCHORS = [
    "brace", "bracket", "gusset", "strut", "reinforcement",
    "clamp", "strap", "tie", "retainer", "fastener",
    "washer", "spacer", "shim", "wedge", "sleeve", "bushing", "bearing",
    "threadlocker", "lock washer", "nylon insert", "set screw",
    "rubber foot", "rubber pad", "felt pad", "anti-slip pad", "grip tape",
    "damper", "damping", "shock absorber", "spring",
    "foam", "sponge", "cushioning", "buffer",
    "lubricant", "grease", "wax",
    "sealant", "epoxy", "adhesive",
    "leveling", "leveling pad",
]


# =========================
# Paths (auto-detect)
# =========================
def _find_data_dir(start_dir: str) -> Tuple[str, str]:
    """
    Return (layout, data_dir)
    layout:
      - "website": <root>/data/chair
      - "proto":   <root>/data
    """
    cur = os.path.abspath(start_dir)
    for _ in range(7):
        cand_dom = os.path.join(cur, "data", "chair")
        cand_data = os.path.join(cur, "data")
        if os.path.isdir(cand_dom):
            return "website", cand_dom
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
# Output junk / pronoun / human bans (shared)
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
}

def _starts_with_bad_determiner(s: str) -> bool:
    return bool(BAD_START_OUT.search(_norm(s)))

def _starts_with_quant_prefix(s: str) -> bool:
    return bool(QUANT_PREFIX_OUT.search(_norm(s)))

def _contains_human_role(s: str) -> bool:
    toks = _norm(s).split()
    return any(t in HUMAN_ROLE_NOUNS for t in toks)


# =========================
# Chair-specific output bans (KEEP, but applied via unified ban pipeline)
# =========================
CHAIR_COMPONENTS = {
    "chair","seat","seats","back","backrest",
    "leg","legs","arm","arms","armrest","armrests",
    "frame","base","foot","feet","cushion","headrest","lumbar",
}

CHAIR_SUBSTITUTE_BAN = {
    "stool","stools","bench","benches","seat","seats","seating",
    "armchair","armchairs","sofa","sofas","couch","couches",
    "footrest","footrests","headrest","headrests",
    "ottoman","ottomans","saddle","saddles","wheelchair","wheelchairs",
}

GENERIC_OBJECT = {"thing","things","object","objects","item","items"}

def _contains_chair_component(phrase: str) -> bool:
    toks = _norm(phrase).split()
    return any(t in CHAIR_COMPONENTS for t in toks)

def _contains_banned_seating_vocab(phrase: str) -> bool:
    toks = _norm(phrase).split()
    return any(t in CHAIR_SUBSTITUTE_BAN for t in toks)

def _is_too_generic(phrase: str) -> bool:
    return _norm(phrase) in GENERIC_OBJECT


# =========================
# Echo-ban (lemma based, like pen v6)
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
# Negative signal extraction (rule + fallback) - chair flavored (POWERED)
# (coverage expanded + weak-signal cleanup; pipeline unchanged)
# =========================
GENERIC_ADJ_STOP = {
    "new","old","other","same","different","first","last","many","few","most","some",
    "large","small","big","high","low","long","short"
}

# NEW: weak adjectives that often pollute NEG centroid (keep it tiny & stable)
WEAK_ADJ_STOP = {
    "little","slight","slightly","minor","smallish","sharp","kind","sort"
}

# NEW: time / discourse fillers that should not become NEG seeds
_TIME_FILLER_PAT = re.compile(
    r"\b("
    r"a\s+few\s+(seconds|minutes|hours|days)|"
    r"after\s+a\s+few\s+(seconds|minutes|hours|days)|"
    r"after\s+a\s+while|"
    r"sometimes|"
    r"every\s+now\s+and\s+then|"
    r"once\s+in\s+a\s+while"
    r")\b",
    re.I
)

# UPGRADED cues: instability + fasteners + mechanisms + casters + locks + noises
ISSUE_VERB_CUE = re.compile(
    r"\b("
    # wobble / instability
    r"wobble|wobbles|wobbling|rock|rocks|rocking|"
    r"tilt|tilts|tilting|lean|leans|leaning|"
    r"unstable|unsteady|"
    # looseness / joints / fasteners
    r"loose|loosens|loosened|loosening|"
    r"rattle|rattles|rattling|"
    r"wiggle|wiggles|wiggling|"
    r"shake|shakes|shaking|"
    r"stripp(ed|ing)|missing|"
    # noises
    r"squeak|squeaks|squeaking|"
    r"creak|creaks|creaking|"
    r"click|clicks|clicking|"
    r"grind|grinds|grinding|"
    r"scrape|scrapes|scraping|"
    # damage / failure
    r"break|breaks|broken|"
    r"crack|cracks|cracked|cracking|"
    r"split|splits|splitting|"
    r"warp|warps|warping|"
    r"bend|bends|bent|bending|"
    r"tear|tears|torn|ripping|rip|"
    r"collapse|collapses|collapsed|"
    r"tip\s*over|tips\s*over|tipping\s*over|"
    # mechanisms (office chair)
    r"jam|jams|jammed|jamming|"
    r"stuck|stick|sticks|sticking|"
    r"sink|sinks|sinking|"
    r"won['’]?t\s+lock|doesn['’]?t\s+lock|"
    r"won['’]?t\s+raise|doesn['’]?t\s+raise|"
    r"won['’]?t\s+lower|doesn['’]?t\s+lower|"
    # movement / casters
    r"slip|slips|slipping|"
    r"slide|slides|sliding|"
    r"roll|rolls|rolling|"
    r"won['’]?t\s+roll|doesn['’]?t\s+roll|"
    # wear
    r"worn|wear|wearing|"
    r")\b",
    re.I
)

# UPGRADED physical patterns: leveling, uneven legs, fasteners, gas lift, tilt lock, casters
PHYS_PATTERNS = [
    re.compile(r"\btoo\s+(?P<x>[a-z][a-z\-]{1,30})\b", re.I),
    re.compile(r"\bnot\s+(?P<x>[a-z][a-z\-]{1,30})\b", re.I),
    re.compile(r"\b(no|poor|bad)\s+(?P<x>[a-z][a-z\-]{1,30})\b", re.I),

    re.compile(r"\b(uneven\s+floor|uneven\s+surface)\b", re.I),
    re.compile(r"\b(not\s+level|unlevel)\b", re.I),
    re.compile(r"\b(one\s+leg\s+shorter|short\s+leg|uneven\s+legs?)\b", re.I),
    re.compile(r"\b(rocks?\s+back\s+and\s+forth|rocking\s+back\s+and\s+forth)\b", re.I),
    re.compile(r"\b(leans?\s+to\s+one\s+side|tilts?\s+to\s+one\s+side)\b", re.I),

    re.compile(r"\b(loose\s+(screw|screws|bolt|bolts|nut|nuts|joint|joints))\b", re.I),
    re.compile(r"\b(missing\s+(screw|screws|bolt|bolts|nut|nuts|washer|washers))\b", re.I),
    re.compile(r"\b(stripped\s+(thread|threads|screw|screws|bolt|bolts))\b", re.I),

    re.compile(r"\b(gas\s+lift\s+sinks?|keeps?\s+sinking)\b", re.I),
    re.compile(r"\b(won['’]?t\s+stay\s+up|won['’]?t\s+hold\s+height)\b", re.I),
    re.compile(r"\b(tilt\s+lock\s+won['’]?t\s+engage|tilt\s+lock\s+fails)\b", re.I),

    re.compile(r"\b(caster\s+stuck|stuck\s+caster|wheel\s+stuck|stuck\s+wheel)\b", re.I),
    re.compile(r"\b(won['’]?t\s+roll|doesn['’]?t\s+roll|hard\s+to\s+roll)\b", re.I),
]

# UPGRADED fallback prototypes: cover office chair mechanisms too
_COMPLAINT_PROTOTYPES = [
    "chair wobbles when you sit down",
    "chair rocks on an uneven floor",
    "chair leans to one side",
    "one chair leg is shorter",
    "chair squeaks when leaning back",
    "chair creaks loudly under weight",
    "chair rattles from a loose joint",
    "chair has a loose screw or bolt",
    "chair base is cracked",
    "chair suddenly collapses",
    "office chair gas lift keeps sinking",
    "chair height won't stay up",
    "tilt lock won't engage",
    "chair wheel is stuck and won't roll",
    "casters don't roll smoothly",
]

def _signal_is_junk(s: str) -> bool:
    """Tiny, stable cleanup to keep NEG seeds meaningful (NOT a domain gate)."""
    s0 = _norm(s)
    if not s0:
        return True
    # remove time/discourse fillers
    if _TIME_FILLER_PAT.search(s0):
        return True
    # drop very generic chunks that are basically determiners + generic noun
    if s0 in {"the floor", "the chair", "a chair", "the leg", "the legs"}:
        return True
    # drop single weak adjectives
    if len(s0.split()) == 1 and s0 in WEAK_ADJ_STOP:
        return True
    return False

def extract_negative_signals_rule(text: str, max_signals: int = 25) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp(text)
    signals: List[str] = []

    # (1) adjectives (drop weak ones)
    for tok in doc:
        if tok.pos_ == "ADJ":
            w = tok.lemma_.lower()
            if w in GENERIC_ADJ_STOP or w in WEAK_ADJ_STOP:
                continue
            if len(w) < 3:
                continue
            signals.append(w)

    # (2) issue cues / verbs
    blob = _norm(text)
    for m in ISSUE_VERB_CUE.finditer(blob):
        signals.append(_norm(m.group(0)))

    # (3) noun chunks that look issue-ish (expanded keywords)
    issue_kw = (
        r"(wobble|rock|tilt|lean|unstable|unsteady|"
        r"loose|rattle|wiggle|shake|"
        r"squeak|creak|click|grind|scrape|"
        r"break|broken|crack|cracked|split|warp|bent|tear|torn|collapse|"
        r"slip|slide|roll|wheel|caster|"
        r"screw|bolt|nut|washer|joint|thread|stripped|missing|"
        r"gas\s+lift|lift|height|lock|tilt\s+lock|swivel|"
        r"floor|level)"
    )
    for chunk in doc.noun_chunks:
        t = _norm(chunk.text)
        if not t:
            continue
        if len(t.split()) > 8:
            continue
        if _signal_is_junk(t):
            continue
        has_adj = any(tk.pos_ == "ADJ" for tk in chunk)
        has_issue_word = bool(re.search(rf"\b{issue_kw}\b", t))
        if has_adj or has_issue_word:
            signals.append(t)

    # (4) regex patterns
    for pat in PHYS_PATTERNS:
        for m in pat.finditer(blob):
            seg = _norm(m.group(0))
            if seg and len(seg) >= 3 and (not _signal_is_junk(seg)):
                signals.append(seg)

    # final cleanup + dedup
    signals = [s for s in signals if not _signal_is_junk(s)]
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
# MMR selection (same as pen)
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
# Bank sanity gate (same as pen FIX #2)
# =========================
_BANK_MAX_TOKENS = 6
_BANK_STOPWORD_RATIO_MAX = 0.45

def _bank_phrase_sane(p: str) -> Tuple[bool, str]:
    """
    Keep bank phrases short & non-encyclopedic.
    NOT a chair-domain filter; just prevent junk phrases.
    """
    p0 = _norm(p)
    if not p0:
        return False, "empty"

    toks = p0.split()
    if len(toks) > _BANK_MAX_TOKENS:
        return False, "too_long"

    if any(ch.isdigit() for ch in p0):
        if not _MM_OK.match(p0.replace(" ", "")) and not _MM_OK.match(p0):
            return False, "has_digits"

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

    if any(t.pos_ in ("VERB", "AUX") for t in doc):
        return False, "has_verb"

    return True, ""


# =========================
# Condition bank expansion + drift (same logic as pen)
# =========================
@dataclass
class ChairResources:
    obj_texts: List[str]
    obj_index: faiss.Index
    bank_phrases: List[str]
    bank_index: faiss.Index
    bank_name: str
    model_name: str = MODEL_NAME_DEFAULT

_RES: Optional[ChairResources] = None

def _load_bank(bank_csv: str, bank_faiss: str, model_name: str) -> Tuple[List[str], faiss.Index]:
    df = pd.read_csv(bank_csv)
    if "phrase" in df.columns:
        phrases = df["phrase"].astype(str).tolist()
    else:
        phrases = df.iloc[:, 0].astype(str).tolist()

    if os.path.exists(bank_faiss):
        idx = faiss.read_index(bank_faiss)
        return phrases, idx

    model = get_model(model_name)
    emb = np.asarray(model.encode(phrases, normalize_embeddings=True, show_progress_bar=True), dtype="float32")
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return phrases, idx

def load_chair_resources(model_name: str = MODEL_NAME_DEFAULT) -> ChairResources:
    global _RES
    if _RES is not None:
        return _RES

    if not os.path.exists(CHAIR_OBJECTS_META):
        raise FileNotFoundError(f"Missing: {CHAIR_OBJECTS_META}")
    if not os.path.exists(CHAIR_OBJECTS_FAISS):
        raise FileNotFoundError(f"Missing: {CHAIR_OBJECTS_FAISS}")

    obj_df = pd.read_csv(CHAIR_OBJECTS_META)
    if "object" in obj_df.columns:
        obj_texts = obj_df["object"].astype(str).tolist()
    elif "term" in obj_df.columns:
        obj_texts = obj_df["term"].astype(str).tolist()
    else:
        raise ValueError("chair_objects_meta.csv must contain column 'object' or 'term'")
    obj_index = faiss.read_index(CHAIR_OBJECTS_FAISS)

    # use indexed condition bank (meta + faiss)
    if not os.path.exists(CHAIR_COND_BANK_META):
        raise FileNotFoundError(f"Missing: {CHAIR_COND_BANK_META}")
    if not os.path.exists(CHAIR_COND_BANK_FAISS):
        raise FileNotFoundError(f"Missing: {CHAIR_COND_BANK_FAISS}")

    bank_phrases, bank_index = _load_bank(
        CHAIR_COND_BANK_META,
        CHAIR_COND_BANK_FAISS,
        model_name
    )
    bank_name = "indexed"

    _RES = ChairResources(
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
    res: ChairResources,
    topn: int = BANK_TOPN_DEFAULT,
) -> Tuple[List[str], List[Dict[str, float]], Dict[str, Any]]:
    """
    Bank expands NEG phrases only (bonus).
    NO chair-domain gate here.
    YES sanity gate here.
    """
    gate_dbg = {
        "bank_available": True,
        "bank_name": res.bank_name,
        "bank_csv": CHAIR_COND_BANK_META,
        "bank_faiss": CHAIR_COND_BANK_FAISS,
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
# EDGE phrase-shape filter (shape-only in opposite mode, same as pen)
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

    # numeric junk ban (strict for chair)
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
# Final selection: top-k closest with diversity caps (same as pen)
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
# Quality ban (same as pen, small)
# =========================
_QUALITY_EXACT_BAN = {"unknown", "none", "misc", "various", "different", "others"}
_QUALITY_BAD_CHARS = re.compile(r"[^\x00-\x7F]")
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

    toks = s0.split()
    if len(toks) == 1:
        w = toks[0]
        if len(w) >= 8 and any(w.endswith(suf) for suf in _QUALITY_BIOMOL_SUFFIX):
            return True, "quality_biomolecule_suffix"

    return False, ""


# =========================
# Unified output-side bans (shared + chair-specific)
# =========================
def _ban_candidate(
    cand: str,
    input_lemmas: set,
    enable_echo_ban: bool = True,
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

    # chair domain bans
    if _contains_chair_component(c0):
        return True, "chair_component"
    if _contains_banned_seating_vocab(c0):
        return True, "chair_substitute"
    if _is_too_generic(c0):
        return True, "generic"

    if enable_echo_ban and _echo_ban_hit(c0, input_lemmas):
        return True, "echo"

    return False, ""


# =========================
# Opposite gate: anti-neg + solution-ish gate (same as pen)
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
        q_rank = np.zeros((384,), dtype="float32")
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

    neg_thr = float(np.quantile(sim_neg, float(neg_sim_q)))
    keep1 = sim_neg <= neg_thr
    dbg["neg_sim_max_used"] = float(neg_thr)

    idx1 = np.where(keep1)[0].tolist()
    dbg["kept_after_anti_neg_gate"] = int(len(idx1))
    if not idx1:
        dbg["notes"].append("anti-neg gate removed all; fallback=keep all")
        idx1 = list(range(len(pool)))

    sim_sol_surv = sim_sol[idx1]
    sol_thr_q = float(np.quantile(sim_sol_surv, float(sol_sim_q))) if len(sim_sol_surv) else 0.0
    sol_thr = max(float(sol_thr_q), float(sol_sim_min))
    dbg["sol_sim_min_used"] = float(sol_thr)

    idx2 = [idx1[j] for j in range(len(idx1)) if float(sim_sol[idx1[j]]) >= sol_thr]
    dbg["kept_after_solution_gate"] = int(len(idx2))

    if not idx2:
        dbg["notes"].append("solution gate removed all; fallback=use anti-neg survivors")
        idx2 = idx1

    score = float(a_sol) * sim_sol - float(b_neg) * sim_neg
    score2 = score[idx2]
    order = np.argsort(-score2)
    ranked = [pool[idx2[i]] for i in order.tolist()]

    dbg["score_minmax"] = [float(np.min(score2)), float(np.max(score2))] if len(score2) else None

    return ranked, dbg, q_rank


# =========================
# Main API (same structure as pen)
# =========================
def get_chair_add(
    text: str,
    debug_json: bool = False,
    opposite_enabled: bool = OPPOSITE_ENABLED_DEFAULT,
) -> Dict[str, Any]:
    res = load_chair_resources(model_name=MODEL_NAME_DEFAULT)
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
                    "objects_meta": CHAIR_OBJECTS_META,
                    "objects_faiss": CHAIR_OBJECTS_FAISS,
                    "cond_bank_csv": CHAIR_COND_BANK_META,
                    "cond_bank_faiss": CHAIR_COND_BANK_FAISS,
                },
            }
        return out

    # 3) retrieve object pool using NEG query
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
        "chair_component": 0,
        "chair_substitute": 0,
        "generic": 0,
        "echo": 0,
        "bad_start": 0,
        "human_role": 0,
        "empty": 0,
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
                "objects_meta": CHAIR_OBJECTS_META,
                "objects_faiss": CHAIR_OBJECTS_FAISS,
                "cond_bank_csv": CHAIR_COND_BANK_META,
                "cond_bank_faiss": CHAIR_COND_BANK_FAISS,
            },
            "policy": {
                "target_topk": TARGET_TOPK,
                "cand_k": CAND_K_DEFAULT,
                "edge_topk": EDGE_TOPK,
                "mmr_lambda": MMR_LAMBDA_DEFAULT,
                "echo_ban": ECHO_BAN_ENABLED_DEFAULT,
                "max_per_head": MAX_PER_HEAD_DEFAULT,
                "bank_policy": "bank expands NEG only (bonus); NO chair-domain gate; YES sanity gate; used if not drifted and kept>=min_bank_to_use; cap=bank_seed_cap",
                "bank_sanity": {
                    "max_tokens": _BANK_MAX_TOKENS,
                    "stopword_ratio_max": _BANK_STOPWORD_RATIO_MAX,
                    "reject_verbs": True,
                },
                "opposite_policy": {
                    "enabled": opposite_enabled,
                    "anchors": "chair repair / stability oriented list (cross-domain allowed)",
                    "neg_gate": f"sim_neg <= quantile(sim_neg, {NEG_SIM_Q_DEFAULT})",
                    "sol_gate": f"sim_sol >= max(quantile(sim_sol_on_survivors, {SOL_SIM_Q_DEFAULT}), {SOL_SIM_MIN_DEFAULT})",
                    "score": f"{A_SOL_DEFAULT}*sim_sol - {B_NEG_DEFAULT}*sim_neg",
                    "mmr_query": "normalize(a*q_sol - b*q_neg)",
                    "edge_filter": "shape-only",
                    "quality_ban": "enabled (tiny)",
                },
            }
        }

    return out


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Describe chair issues (English).")
    ap.add_argument("--debug-json", action="store_true", help="Include debug section in output.")
    ap.add_argument("--no-opposite", action="store_true", help="Disable opposite gate (for debugging).")
    args = ap.parse_args()

    out = get_chair_add(
        text=args.text,
        debug_json=args.debug_json,
        opposite_enabled=(not args.no_opposite),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()