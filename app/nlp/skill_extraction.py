"""
skill_extraction.py — Extraction des compétences depuis un texte de CV
Utilise skills_list.json comme référentiel de compétences
"""

import json
from pathlib import Path
from app.config import SKILLS_FILE
from app.utils.logger import logger


# ── Chargement du référentiel ─────────────────────────────────────────────────

def load_skills_db() -> dict:
    """Charge le fichier skills_list.json"""
    try:
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            skills = json.load(f)
        total = sum(len(v) for v in skills.values())
        logger.info(f"Skills DB chargée : {len(skills)} catégories, {total} compétences")
        return skills
    except Exception as e:
        logger.error(f"Erreur chargement skills_list.json : {e}")
        return {}


# ── Extraction principale ─────────────────────────────────────────────────────

def extract_skills(text: str, skills_db: dict = None) -> dict:
    """
    Extrait les compétences trouvées dans un texte.
    Retourne un dict par catégorie avec les compétences détectées.

    Exemple de retour :
    {
        "langages":      ["python", "sql"],
        "frameworks_ml": ["scikit-learn", "bert"],
        "nlp":           ["nlp", "embeddings"],
        ...
    }
    """
    if skills_db is None:
        skills_db = load_skills_db()

    if not text:
        return {}

    text_lower = text.lower()
    found = {}

    for category, skills_list in skills_db.items():
        matched = []
        for skill in skills_list:
            # Recherche exacte du mot/groupe de mots
            if skill.lower() in text_lower:
                matched.append(skill)
        if matched:
            found[category] = matched

    total_found = sum(len(v) for v in found.values())
    logger.debug(f"Compétences extraites : {total_found} dans {len(found)} catégories")
    return found


def extract_skills_flat(text: str, skills_db: dict = None) -> list[str]:
    """
    Version à plat — retourne une simple liste de compétences trouvées.
    Utile pour le calcul du score.
    """
    found = extract_skills(text, skills_db)
    return [skill for skills in found.values() for skill in skills]


# ── Matching skills CV ↔ Offre ────────────────────────────────────────────────

def match_skills(cv_text: str, job_text: str, skills_db: dict = None) -> dict:
    """
    Compare les compétences d'un CV avec celles d'une offre d'emploi.

    Retourne :
    {
        "cv_skills":      [...],   # compétences du CV
        "job_skills":     [...],   # compétences demandées dans l'offre
        "matched":        [...],   # compétences en commun
        "missing":        [...],   # compétences manquantes dans le CV
        "match_rate":     0.75,    # taux de correspondance (0-1)
    }
    """
    if skills_db is None:
        skills_db = load_skills_db()

    cv_skills  = set(extract_skills_flat(cv_text,  skills_db))
    job_skills = set(extract_skills_flat(job_text, skills_db))

    if not job_skills:
        logger.warning("Aucune compétence détectée dans l'offre d'emploi")
        return {
            "cv_skills":  list(cv_skills),
            "job_skills": [],
            "matched":    [],
            "missing":    [],
            "match_rate": 0.0,
        }

    matched    = cv_skills & job_skills
    missing    = job_skills - cv_skills
    match_rate = len(matched) / len(job_skills)

    logger.info(
        f"Skills matching : {len(matched)}/{len(job_skills)} "
        f"({match_rate:.0%}) — manquantes : {len(missing)}"
    )

    return {
        "cv_skills":  sorted(cv_skills),
        "job_skills": sorted(job_skills),
        "matched":    sorted(matched),
        "missing":    sorted(missing),
        "match_rate": round(match_rate, 4),
    }