"""
similarity.py — Calcul de similarité cosinus entre CV et offre d'emploi
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.nlp.embeddings import get_embedding, get_embeddings_batch
from app.utils.logger import logger


# ── Similarité simple (1 CV ↔ 1 offre) ───────────────────────────────────────

def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Calcule la similarité cosinus entre deux textes.
    Retourne un score entre 0.0 et 1.0.

    Usage :
        score = compute_similarity(cv_text, job_text)
        # score = 0.87 → très bonne correspondance
    """
    vec_a = get_embedding(text_a).reshape(1, -1)
    vec_b = get_embedding(text_b).reshape(1, -1)

    score = float(cosine_similarity(vec_a, vec_b)[0][0])
    score = max(0.0, min(1.0, score))   # clamp entre 0 et 1

    logger.debug(f"Similarité cosinus : {score:.4f}")
    return score


# ── Similarité batch (N CVs ↔ 1 offre) ───────────────────────────────────────

def compute_similarity_batch(cv_texts: list[str], job_text: str) -> list[float]:
    """
    Calcule la similarité cosinus entre plusieurs CVs et une offre d'emploi.
    Beaucoup plus efficace que d'appeler compute_similarity() en boucle.

    Retourne une liste de scores [0.0 → 1.0] dans le même ordre que cv_texts.

    Usage :
        scores = compute_similarity_batch(["cv1...", "cv2...", ...], job_description)
    """
    if not cv_texts:
        return []

    logger.info(f"Calcul similarité batch : {len(cv_texts)} CVs vs 1 offre")

    # Encoder tous les CVs en batch + l'offre d'emploi
    cv_vectors  = get_embeddings_batch(cv_texts)          # (N x 384)
    job_vector  = get_embedding(job_text).reshape(1, -1)  # (1 x 384)

    # Calcul cosine similarity : résultat shape (N x 1)
    scores = cosine_similarity(cv_vectors, job_vector).flatten()

    # Clamp entre 0 et 1 + conversion en liste Python
    scores = [float(max(0.0, min(1.0, s))) for s in scores]

    logger.info(f"Scores : min={min(scores):.3f} | max={max(scores):.3f} | moy={sum(scores)/len(scores):.3f}")
    return scores


# ── Conversion score 0-1 → pourcentage lisible ───────────────────────────────

def score_to_percent(score: float) -> int:
    """Convertit un score cosinus [0-1] en pourcentage entier [0-100]."""
    return round(score * 100)


def interpret_score(score: float) -> str:
    """
    Retourne une interprétation textuelle du score.
    Utilisé pour l'affichage dans l'interface Streamlit.
    """
    pct = score_to_percent(score)
    if pct >= 75:
        return "🟢 Excellent match"
    elif pct >= 55:
        return "🟡 Bon match"
    elif pct >= 35:
        return "🟠 Match moyen"
    else:
        return "🔴 Faible match"