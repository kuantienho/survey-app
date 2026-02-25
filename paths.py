# paths.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

def _p(*parts):
    return os.path.join(*parts)

# -------------------------
# PEN
# -------------------------
PEN_DIR = _p(DATA_DIR, "pen")
PEN_OBJECTS_FAISS = _p(PEN_DIR, "pen_objects.faiss")
PEN_OBJECTS_META  = _p(PEN_DIR, "pen_objects_meta.csv")
PEN_COND_BANK_FAISS = _p(PEN_DIR, "pen_condition_bank.faiss")
PEN_COND_BANK_META  = _p(PEN_DIR, "pen_condition_bank_meta.csv")

# -------------------------
# UMBRELLA
# -------------------------
UMBRELLA_DIR = _p(DATA_DIR, "umbrella")
UMBRELLA_OBJECTS_FAISS = _p(UMBRELLA_DIR, "umbrella_objects.faiss")
UMBRELLA_OBJECTS_META  = _p(UMBRELLA_DIR, "umbrella_objects_meta.csv")
UMBRELLA_COND_BANK_FAISS = _p(UMBRELLA_DIR, "umbrella_condition_bank.faiss")
UMBRELLA_COND_BANK_META  = _p(UMBRELLA_DIR, "umbrella_condition_bank_meta.csv")

# -------------------------
# CHAIR
# -------------------------
CHAIR_DIR = _p(DATA_DIR, "chair")
CHAIR_OBJECTS_FAISS = _p(CHAIR_DIR, "chair_objects.faiss")
CHAIR_OBJECTS_META  = _p(CHAIR_DIR, "chair_objects_meta.csv")
CHAIR_COND_BANK_FAISS = _p(CHAIR_DIR, "chair_condition_bank.faiss")
CHAIR_COND_BANK_META  = _p(CHAIR_DIR, "chair_condition_bank_meta.csv")

# -------------------------
# PSU (Power Supply)
# -------------------------
PSU_DIR = _p(DATA_DIR, "psu")
PSU_OBJECTS_FAISS = _p(PSU_DIR, "psu_objects.faiss")
PSU_OBJECTS_META  = _p(PSU_DIR, "psu_objects_meta.csv")
PSU_COND_BANK_FAISS = _p(PSU_DIR, "psu_condition_bank.faiss")
PSU_COND_BANK_META  = _p(PSU_DIR, "psu_condition_bank_meta.csv")