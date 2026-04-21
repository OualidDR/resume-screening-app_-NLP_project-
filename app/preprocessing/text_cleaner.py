"""
text_cleaner.py — Nettoyage et normalisation du texte des CVs
Pipeline : urls → mentions → ponctuation → espaces → minuscules
"""

import re
from app.utils.logger import logger


# ── Pipeline de nettoyage ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Nettoie un texte brut extrait d'un CV.
    Retourne un texte normalisé prêt pour les embeddings.
    """
    if not text or not isinstance(text, str):
        return ""

    original_len = len(text)

    # 1. Supprimer les URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 2. Supprimer les emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 3. Supprimer les mentions Twitter/LinkedIn (@user, #hashtag)
    text = re.sub(r"[@#]\S+", " ", text)

    # 4. Supprimer les caractères non-ASCII (emojis, symboles exotiques)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 5. Supprimer la ponctuation sauf les tirets (utiles pour "data-science")
    text = re.sub(r"[^\w\s\-]", " ", text)

    # 6. Supprimer les nombres isolés (ex: numéros de téléphone)
    text = re.sub(r"\b\d{4,}\b", " ", text)

    # 7. Normaliser les espaces multiples / sauts de ligne
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Mettre en minuscules
    text = text.lower()

    logger.debug(f"Nettoyage : {original_len} → {len(text)} caractères")
    return text


def clean_for_display(text: str, max_chars: int = 500) -> str:
    """
    Version légère pour affichage dans l'UI Streamlit.
    Conserve la ponctuation, juste normalise les espaces.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


# ── Nettoyage batch (pour le dataset CSV) ────────────────────────────────────

def clean_records(records: list[dict]) -> list[dict]:
    """
    Applique clean_text() sur une liste de records chargés depuis le CSV.
    Chaque record doit avoir une clé 'text'.
    Retourne les records avec un champ 'clean_text' ajouté.
    """
    cleaned = []
    for record in records:
        record["clean_text"] = clean_text(record.get("text", ""))
        cleaned.append(record)

    logger.info(f"Nettoyage batch terminé : {len(cleaned)} CVs traités")
    return cleaned