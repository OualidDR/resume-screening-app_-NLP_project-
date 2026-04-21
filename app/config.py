"""
config.py — Configuration globale du projet
Toutes les constantes et paramètres sont centralisés ici.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Chemins du projet ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # racine du projet
DATA_DIR    = BASE_DIR / "data"
CV_DIR      = DATA_DIR / "cvs"
JOB_DIR     = DATA_DIR / "jobs"
SKILLS_FILE = DATA_DIR / "skills_list.json"
ARTIFACTS   = BASE_DIR / "artifacts"
CACHE_DIR   = ARTIFACTS / "embeddings_cache"

# Créer les dossiers manquants automatiquement
for _dir in [CV_DIR, JOB_DIR, ARTIFACTS, CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Modèle BERT ───────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "all-MiniLM-L6-v2"   # léger, rapide, multilingue acceptable
# Pour du français pur, remplacer par : "paraphrase-multilingual-MiniLM-L12-v2"


# ── Scoring ───────────────────────────────────────────────────────────────────
# Pondération du score final (doit totaliser 1.0)
WEIGHTS = {
    "semantic_similarity": 0.60,   # similarité BERT globale CV ↔ offre
    "skills_match":        0.40,   # compétences extraites présentes dans l'offre
}

# Seuils de classement (score sur 100)
SCORE_THRESHOLDS = {
    "excellent": 75,
    "bon":       55,
    "moyen":     35,
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
MAX_TEXT_LENGTH = 10_000   # caractères max conservés par CV
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")