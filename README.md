# 📄 Resume Screening App — NLP & Modèles Sémantiques

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
> Application intelligente de présélection de CV utilisant l'IA (BERT / Sentence-Transformers) et la similarité cosinus pour automatiser le matching sémantique entre profils candidats et offres d'emploi.

---

## 📑 Table des matières
- [Fonctionnalités](#-fonctionnalités)
- [ Installation](#-installation)
- [ Architecture](#️-architecture)
- [ Stack Technique](#️-stack_technique)
- [Pipeline](#️-Pipeline)


---
## ✨ Fonctionnalités

| Catégorie | Détails |
|:----------|:--------|
| 📥 **Extraction Multi-Format** | Parsing automatique des fichiers PDF (`PyPDF2`) et DOCX (`python-docx`) avec préservation de la structure. |
| 🧠 **Embeddings Sémantiques** | Vectorisation contextuelle via `all-MiniLM-L6-v2` pour une compréhension fine du langage naturel. |
| 📊 **Scoring Hybride** | Score de pertinence combiné : `70% similarité sémantique` + `30% matching de compétences techniques`. |
| 🎯 **Classement Automatique** | Tri décroissant des candidats avec visualisation des forces/faiblesses par rapport à l'offre. |
| 🖥️ **Interface Streamlit** | Application web interactive : upload drag & drop, zone de saisie d'offre, mode sombre/clair. |
| 🪵 **Logging & Cache** | Suivi structuré avec `loguru` et mise en cache locale des embeddings pour accélérer les analyses. |

---

## 🚀 Installation

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd resume-screening-app

# 2. Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
```

## ▶Lancement

```bash
streamlit run run.py
```

---

## Architecture

```
resume-screening-app/
├── app/
│   ├── config.py              # Configuration globale
│   ├── main.py                # Interface Streamlit
│   ├── preprocessing/         # Extraction et nettoyage texte
│   ├── nlp/                   # BERT embeddings + similarité
│   ├── scoring/               # Score et classement
│   └── utils/                 # Logger, utilitaires
├── data/
│   ├── skills_list.json       # Base de compétences
│   ├── cvs/                   # CVs de test
│   └── jobs/                  # Offres d'emploi
├── artifacts/                 # Cache embeddings
├── notebooks/                 # Expérimentation Colab
├── tests/                     # Tests unitaires
├── requirements.txt
└── run.py
```

## Stack_technique

| Composant | Technologie |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Similarité | Cosine similarity (`scikit-learn`) |
| Extraction PDF | `PyPDF2` |
| Extraction DOCX | `python-docx` |
| Interface | `Streamlit` |
| Logging | `loguru` |

---

## Pipeline

```
CV (PDF/DOCX) ──► Extraction texte ──► Nettoyage
                                            │
Offre d'emploi ─────────────────────────────┤
                                            ▼
                              Embeddings BERT (sentence-transformers)
                                            │
                                            ▼
                              Cosine Similarity + Skills Matching
                                            │
                                            ▼
                              Score final + Classement candidats
```