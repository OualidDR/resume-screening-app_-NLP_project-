"""
ranking.py — Classement des candidats par score de pertinence
"""

import pandas as pd
from app.utils.logger import logger


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """
    Trie les candidats par score décroissant.

    Chaque candidat doit avoir au minimum :
    {
        "name":         "candidat_1.pdf",
        "final_score":  0.78,
        "percent":      78,
        "label":        "🟢 Excellent match — 78%",
        "skills":       {...}
    }

    Retourne la liste triée avec le rang ajouté.
    """
    if not candidates:
        logger.warning("Aucun candidat à classer")
        return []

    # Tri par score décroissant
    ranked = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)

    # Ajouter le rang
    for i, candidate in enumerate(ranked, start=1):
        candidate["rank"] = i

    logger.info(f"Classement terminé : {len(ranked)} candidats")
    logger.info(f"  🥇 1er : {ranked[0].get('name', '?')} — {ranked[0].get('percent', 0)}%")

    return ranked


def to_dataframe(ranked_candidates: list[dict]) -> pd.DataFrame:
    """
    Convertit la liste des candidats classés en DataFrame pour Streamlit.
    Retourne un DataFrame propre avec les colonnes essentielles.
    """
    if not ranked_candidates:
        return pd.DataFrame()

    rows = []
    for c in ranked_candidates:
        skills = c.get("skills", {})
        rows.append({
            "Rang":              c.get("rank", "-"),
            "Candidat":          c.get("name", "Inconnu"),
            "Score":             f"{c.get('percent', 0)}%",
            "Niveau":            c.get("label", "-"),
            "Similarité BERT":   f"{round(c.get('semantic_similarity', 0) * 100)}%",
            "Skills matchées":   f"{len(skills.get('matched', []))} / {len(skills.get('job_skills', []))}",
            "Compétences":       ", ".join(skills.get("matched", [])) or "—",
            "Manquantes":        ", ".join(skills.get("missing", [])) or "—",
        })

    df = pd.DataFrame(rows)
    logger.info(f"DataFrame créé : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df


def get_top_n(ranked_candidates: list[dict], n: int = 5) -> list[dict]:
    """Retourne les N meilleurs candidats."""
    return ranked_candidates[:n]