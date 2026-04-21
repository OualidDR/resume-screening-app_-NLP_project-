# 📄 Resume Screening App — NLP & Modèles Sémantiques

Application intelligente de présélection de CV utilisant BERT et la similarité cosinus.

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

## ▶️ Lancement

```bash
streamlit run run.py
```

---

## 🏗️ Architecture

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

## 🔧 Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Similarité | Cosine similarity (`scikit-learn`) |
| Extraction PDF | `PyPDF2` |
| Extraction DOCX | `python-docx` |
| Interface | `Streamlit` |
| Logging | `loguru` |

---

## 📊 Pipeline

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