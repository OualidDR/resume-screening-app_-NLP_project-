"""
scorer.py — Calcul du score final de pertinence d'un CV par rapport à une offre
Score final = 60% similarité sémantique BERT + 40% matching compétences
"""

from app.config import WEIGHTS, SCORE_THRESHOLDS
from app.nlp.skill_extraction import match_skills, load_skills_db
from app.utils.logger import logger


# ── Score final ───────────────────────────────────────────────────────────────

def compute_score(
    semantic_similarity: float,
    skills_match_rate: float,
) -> dict:
    """
    Calcule le score final pondéré.

    Args:
        semantic_similarity : score cosinus BERT [0-1]
        skills_match_rate   : taux de compétences matchées [0-1]

    Retourne un dict avec le score détaillé.
    """
    w_sem   = WEIGHTS["semantic_similarity"]   # 0.60
    w_skills = WEIGHTS["skills_match"]          # 0.40

    final_score = (semantic_similarity * w_sem) + (skills_match_rate * w_skills)
    final_score = round(min(1.0, max(0.0, final_score)), 4)
    percent     = round(final_score * 100)

    # Niveau de correspondance
    if percent >= SCORE_THRESHOLDS["excellent"]:
        level = "Excellent"
        emoji = "🟢"
    elif percent >= SCORE_THRESHOLDS["bon"]:
        level = "Bon"
        emoji = "🟡"
    elif percent >= SCORE_THRESHOLDS["moyen"]:
        level = "Moyen"
        emoji = "🟠"
    else:
        level = "Faible"
        emoji = "🔴"

    return {
        "final_score":          final_score,
        "percent":              percent,
        "level":                level,
        "emoji":                emoji,
        "semantic_similarity":  round(semantic_similarity, 4),
        "skills_match_rate":    round(skills_match_rate, 4),
        "label":                f"{emoji} {level} match — {percent}%",
    }


def score_cv(cv_text: str, job_text: str, semantic_similarity: float) -> dict:
    """
    Score complet d'un CV par rapport à une offre.
    Combine similarité BERT (déjà calculée) + matching compétences.

    Args:
        cv_text              : texte nettoyé du CV
        job_text             : texte nettoyé de l'offre d'emploi
        semantic_similarity  : score cosinus déjà calculé par similarity.py

    Retourne le score final + détails skills.
    """
    skills_db = load_skills_db()

    # Matching compétences
    skills_result = match_skills(cv_text, job_text, skills_db)
    skills_rate   = skills_result["match_rate"]

    # Score final
    score = compute_score(semantic_similarity, skills_rate)

    # Enrichir avec les détails skills
    score["skills"] = skills_result

    logger.info(
        f"Score CV : {score['percent']}% ({score['level']}) — "
        f"BERT={semantic_similarity:.2f} | Skills={skills_rate:.2f}"
    )

    return score