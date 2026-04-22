"""
embeddings.py — Chargement du modèle BERT et génération d'embeddings
Modèle : sentence-transformers/all-MiniLM-L6-v2
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import BERT_MODEL_NAME, CACHE_DIR
from app.utils.logger import logger


# ── Singleton : on charge le modèle une seule fois ────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Charge le modèle BERT une seule fois (pattern Singleton).
    Les appels suivants retournent l'instance déjà en mémoire.
    """
    global _model
    if _model is None:
        logger.info(f"Chargement du modèle BERT : '{BERT_MODEL_NAME}' ...")
        _model = SentenceTransformer(BERT_MODEL_NAME)
        logger.info("Modèle BERT chargé avec succès ✓")
    return _model


# ── Génération d'embeddings ───────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """
    Génère l'embedding d'un texte unique.
    Retourne un vecteur numpy de dimension 384 (MiniLM).

    Usage :
        vec = get_embedding("Data scientist with Python skills")
    """
    if not text or not text.strip():
        logger.warning("Texte vide reçu — retour d'un vecteur zéro")
        return np.zeros(384)

    model  = get_model()
    vector = model.encode(text, convert_to_numpy=True)
    logger.debug(f"Embedding généré : shape={vector.shape}")
    return vector


def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Génère les embeddings pour une liste de textes (traitement par batch).
    Beaucoup plus rapide que d'appeler get_embedding() en boucle.

    Retourne une matrice numpy (N x 384).

    Usage :
        matrix = get_embeddings_batch(["cv1 text", "cv2 text", ...])
    """
    if not texts:
        return np.zeros((0, 384))

    model   = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info(f"Batch embeddings : {len(texts)} textes → shape={vectors.shape}")
    return vectors


# ── Cache (optionnel) ─────────────────────────────────────────────────────────

def save_embeddings(vectors: np.ndarray, name: str) -> Path:
    """Sauvegarde les embeddings dans artifacts/embeddings_cache/"""
    path = CACHE_DIR / f"{name}.npy"
    np.save(path, vectors)
    logger.info(f"Embeddings sauvegardés : {path}")
    return path


def load_embeddings(name: str) -> np.ndarray | None:
    """Charge les embeddings depuis le cache si disponibles."""
    path = CACHE_DIR / f"{name}.npy"
    if path.exists():
        vectors = np.load(path)
        logger.info(f"Embeddings chargés depuis cache : {path} — shape={vectors.shape}")
        return vectors
    logger.warning(f"Cache introuvable : {path}")
    return None