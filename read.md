# 📄 Resume Screening App — NLP & Modèles Sémantiques

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
> Application intelligente de présélection de CV utilisant l'IA (BERT / Sentence-Transformers) et la similarité cosinus pour automatiser le matching sémantique entre profils candidats et offres d'emploi.

---

## 📑 Table des matières
- [✨ Fonctionnalités](#-fonctionnalités)
- [🛠️ Stack Technique](#️-stack-technique)
- [🚀 Installation](#-installation)
- [▶️ Utilisation](#️-utilisation)
- [🏗️ Architecture & Pipeline](#️-architecture--pipeline)
- [⚙️ Configuration](#️-configuration)
- [🧪 Tests](#-tests)
- [🤝 Contribution](#-contribution)
- [📄 Licence](#-licence)
- [📬 Contact](#-contact)
- [🙏 Remerciements](#-remerciements)

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

## 🛠️ Stack Technique

| Domaine | Bibliothèques / Outils |
|:--------|:-----------------------|
| **Langage** | Python `3.9+` |
| **NLP & IA** | `sentence-transformers`, `transformers`, `scikit-learn`, `torch` |
| **Traitement Fichiers** | `PyPDF2==3.0.1`, `python-docx==1.1.2` |
| **Interface Web** | `Streamlit==1.35.0` |
| **Data & Utils** | `pandas`, `numpy`, `loguru==0.7.2`, `python-dotenv` |
| **Tests** | `pytest` |

---

## 🚀 Installation

### Prérequis
- Python `3.9` ou supérieur
- `pip` et `venv` installés
- `4 Go+` de RAM recommandés (pour l'inférence BERT)

### Étapes
```bash
# 1. Cloner le dépôt
git clone https://github.com/OualidDR/resume-screening-app_-NLP_project-.git
cd resume-screening-app_-NLP_project-

# 2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env  # (si disponible, sinon créer manuellement)

from app.nlp.embedding import generate_embedding
from app.scoring.calculator import compute_relevance_score

cv_text = "Texte extrait du candidat..."
job_text = "Description du poste..."

cv_emb = generate_embedding(cv_text)
job_emb = generate_embedding(job_text)

score = compute_relevance_score(
    cv_emb, job_emb,
    cv_skills=["Python", "NLP", "FastAPI"],
    required_skills=["Python", "Machine Learning"]
)
print(f"Score de pertinence : {score:.2%}")
```
```bash
resume-screening-app_-NLP_project-/
├── 📁 app/
│   ├── config.py              # Configuration centralisée
│   ├── main.py                # Interface Streamlit
│   ├── 📁 preprocessing/      # Extraction & nettoyage texte
│   ├── 📁 nlp/                # Embeddings & similarité
│   ├── 📁 scoring/            # Calcul de score & classement
│   └── 📁 utils/              # Logging & helpers
├── 📁 data/
│   ├── skills_list.json       # Base de compétences de référence
│   ├── cvs/                   # CVs de démonstration
│   └── jobs/                  # Offres de démonstration
├── 📁 artifacts/              # Cache des embeddings (gitignored)
├── 📁 notebooks/              # Expérimentations & prototypes
├── 📁 tests/                  # Tests unitaires
├── requirements.txt
└── run.py
```
```bash
[CV PDF/DOCX] ──► [TextExtractor] ──► [TextCleaner]
                                              │
[Offre d'emploi] ─────────────────────────────┤
                                              ▼
                            [SentenceTransformer Embedding]
                                              │
                                              ▼
                    ┌─────────────────────────────────────┐
                    │  Hybrid Scoring Engine:             │
                    │  • Cosine Similarity (70%)          │
                    │  • Skills Keyword Matching (30%)    │
                    └─────────────────────────────────────┘
                                              │
                                              ▼
                            [Ranker] ──► [Streamlit UI]
```