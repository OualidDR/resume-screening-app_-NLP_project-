"""
extractor.py — Extraction de texte depuis différents formats
Formats supportés : PDF, DOCX, TXT, CSV (dataset Kaggle)
"""

import csv
import PyPDF2
import docx
import pandas as pd
from pathlib import Path
from app.utils.logger import logger
from app.config import SUPPORTED_EXTENSIONS, MAX_TEXT_LENGTH


# ── PDF ───────────────────────────────────────────────────────────────────────

def extract_from_pdf(file_path: str | Path) -> str:
    """Extrait le texte d'un fichier PDF page par page."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"PDF extrait : {Path(file_path).name} ({len(text)} caractères)")
    except Exception as e:
        logger.error(f"Erreur extraction PDF '{file_path}' : {e}")
    return text[:MAX_TEXT_LENGTH]


# ── DOCX ──────────────────────────────────────────────────────────────────────

def extract_from_docx(file_path: str | Path) -> str:
    """Extrait le texte d'un fichier Word (.docx)."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        logger.info(f"DOCX extrait : {Path(file_path).name} ({len(text)} caractères)")
    except Exception as e:
        logger.error(f"Erreur extraction DOCX '{file_path}' : {e}")
    return text[:MAX_TEXT_LENGTH]


# ── TXT ───────────────────────────────────────────────────────────────────────

def extract_from_txt(file_path: str | Path) -> str:
    """Extrait le texte d'un fichier .txt (UTF-8 ou latin-1)."""
    text = ""
    try:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        logger.info(f"TXT extrait : {Path(file_path).name} ({len(text)} caractères)")
    except Exception as e:
        logger.error(f"Erreur extraction TXT '{file_path}' : {e}")
    return text[:MAX_TEXT_LENGTH]


# ── CSV (dataset Kaggle : ID, Category, Feature) ──────────────────────────────

def load_dataset_csv(file_path: str | Path) -> list[dict]:
    """
    Charge le dataset Kaggle (colonnes : ID, Category, Feature).
    Retourne une liste de dicts : [{"id": ..., "category": ..., "text": ...}, ...]
    """
    records = []
    try:
        df = pd.read_csv(file_path)

        # Normaliser les noms de colonnes (minuscules, sans espaces)
        df.columns = [c.strip().lower() for c in df.columns]

        # Vérifier que les colonnes attendues existent
        required = {"category", "feature"}
        if not required.issubset(set(df.columns)):
            logger.error(f"Colonnes manquantes dans le CSV. Trouvées : {list(df.columns)}")
            return records

        for _, row in df.iterrows():
            records.append({
                "id":       str(row.get("id", "")),
                "category": str(row.get("category", "")).strip(),
                "text":     str(row.get("feature", ""))[:MAX_TEXT_LENGTH],
            })

        logger.info(f"Dataset chargé : {len(records)} CVs depuis '{Path(file_path).name}'")

    except Exception as e:
        logger.error(f"Erreur chargement CSV '{file_path}' : {e}")

    return records


# ── Dispatcher universel ──────────────────────────────────────────────────────

def extract_text(file_path: str | Path) -> str:
    """
    Point d'entrée unique : détecte l'extension et appelle le bon extracteur.
    Usage : text = extract_text("mon_cv.pdf")
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Format non supporté : '{ext}'. "
            f"Formats acceptés : {SUPPORTED_EXTENSIONS}"
        )

    extractors = {
        ".pdf":  extract_from_pdf,
        ".docx": extract_from_docx,
        ".txt":  extract_from_txt,
    }

    return extractors[ext](path)