"""
main.py — Interface Streamlit pour le Resume Screening App
Pipeline : Upload CVs → Offre d'emploi → Score → Classement
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from app.preprocessing.extractor import extract_text, load_dataset_csv
from app.preprocessing.text_cleaner import clean_text
from app.nlp.skill_extraction import match_skills, load_skills_db
from app.scoring.scorer import compute_score
from app.scoring.ranking import rank_candidates, to_dataframe

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d2137 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 179, 237, 0.2);
}

.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #90cdf4;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #1a365d;
}

.metric-card .label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* Score badge */
.score-excellent { color: #276749; background: #c6f6d5; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
.score-bon       { color: #744210; background: #fefcbf; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
.score-moyen     { color: #7b341e; background: #feebc8; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
.score-faible    { color: #742a2a; background: #fed7d7; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a365d;
    border-left: 4px solid #3182ce;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Skill tags */
.skill-tag {
    display: inline-block;
    background: #ebf8ff;
    color: #2b6cb0;
    border: 1px solid #bee3f8;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
    font-weight: 500;
}

.skill-tag-missing {
    background: #fff5f5;
    color: #c53030;
    border-color: #fed7d7;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f23;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #90cdf4 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 Resume Screening App</h1>
    <p>Analyse intelligente de CVs par NLP · BERT Semantic Matching · Scoring automatique</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    mode = st.radio(
        "Mode d'analyse",
        ["📁 Upload CVs (PDF/DOCX)", "📊 Dataset CSV"],
        help="Choisir la source des CVs à analyser"
    )

    st.markdown("---")
    st.markdown("## 🎛️ Paramètres scoring")

    w_bert = st.slider("Poids similarité BERT", 0.0, 1.0, 0.6, 0.05)
    w_skills = round(1.0 - w_bert, 2)
    st.info(f"Poids compétences : **{w_skills}**")

    top_n = st.slider("Top N candidats à afficher", 5, 50, 10)

    st.markdown("---")
    st.markdown("### 📖 Pipeline")
    st.markdown("""
    1. 📤 Upload CVs
    2. ✍️ Offre d'emploi
    3. 🧹 Preprocessing
    4. 🤖 BERT Embeddings
    5. 🎯 Scoring
    6. 🏆 Classement
    """)


# ── Zone principale ───────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── Colonne gauche : Input CVs ────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">📤 Source des CVs</div>', unsafe_allow_html=True)

    cv_records = []

    if "Dataset CSV" in mode:
        uploaded_csv = st.file_uploader(
            "Upload le dataset CSV (ID, Category, Feature)",
            type=["csv"],
            help="Format Kaggle : colonnes ID, Category, Feature"
        )
        if uploaded_csv:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_csv.read())
                tmp_path = tmp.name
            records = load_dataset_csv(tmp_path)
            os.unlink(tmp_path)

            # Limiter pour la démo
            max_cvs = st.slider("Nombre de CVs à analyser", 10, min(500, len(records)), 50)
            cv_records = records[:max_cvs]
            st.success(f"✅ **{len(cv_records)} CVs** chargés depuis le dataset")

            # Aperçu
            if st.checkbox("Aperçu du dataset"):
                df_preview = pd.DataFrame(cv_records[:5])[["id", "category", "text"]]
                df_preview["text"] = df_preview["text"].str[:100] + "..."
                st.dataframe(df_preview, use_container_width=True)

    else:
        uploaded_files = st.file_uploader(
            "Upload des CVs (PDF ou DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            import tempfile, os
            for f in uploaded_files:
                ext = f.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                text = extract_text(tmp_path)
                os.unlink(tmp_path)
                cv_records.append({
                    "id":       f.name,
                    "category": "Upload",
                    "text":     text,
                })
            st.success(f"✅ **{len(cv_records)} CVs** uploadés")


# ── Colonne droite : Offre d'emploi ──────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">✍️ Offre d\'emploi</div>', unsafe_allow_html=True)

    job_examples = {
        "-- Choisir un exemple --": "",
        "Data Scientist": "We are looking for a Data Scientist with strong experience in Python, machine learning and NLP. Required: pandas, scikit-learn, PyTorch, SQL, BERT, transformers.",
        "Finance Analyst": "Financial Analyst with Excel, VBA, Bloomberg expertise. Experience in financial modeling, risk management and reporting. Python for automation appreciated.",
        "Software Engineer": "Backend software engineer with Python or Java. Experience with Docker, Kubernetes, PostgreSQL, REST APIs, Git and CI/CD pipelines.",
        "HR Manager": "HR Manager with recruitment, onboarding and employee relations experience. Strong communication, leadership and team management skills.",
    }

    selected = st.selectbox("Charger un exemple", list(job_examples.keys()))
    job_text = st.text_area(
        "Description du poste",
        value=job_examples[selected],
        height=200,
        placeholder="Collez ici la description du poste..."
    )

    if job_text:
        skills_db  = load_skills_db()
        from app.nlp.skill_extraction import extract_skills_flat
        job_skills = extract_skills_flat(job_text, skills_db)
        if job_skills:
            st.markdown("**Compétences détectées dans l'offre :**")
            tags = " ".join([f'<span class="skill-tag">{s}</span>' for s in job_skills])
            st.markdown(tags, unsafe_allow_html=True)


# ── Bouton Analyser ───────────────────────────────────────────────────────────
st.markdown("---")

can_analyze = len(cv_records) > 0 and len(job_text.strip()) > 0
btn = st.button(
    "🚀 Lancer l'analyse",
    disabled=not can_analyze,
    use_container_width=True,
    type="primary",
)

if not can_analyze:
    st.info("👆 Upload des CVs et renseigne une offre d'emploi pour lancer l'analyse.")


# ── Analyse ───────────────────────────────────────────────────────────────────
if btn and can_analyze:

    skills_db  = load_skills_db()
    job_clean  = clean_text(job_text)
    candidates = []

    progress = st.progress(0, text="Analyse en cours...")
    total    = len(cv_records)

    for i, record in enumerate(cv_records):
        cv_clean = clean_text(record["text"])

        # Skills matching (local, sans BERT)
        skills_result = match_skills(cv_clean, job_clean, skills_db)

        # Similarité sémantique — tentative BERT, fallback TF-IDF
        try:
            from app.nlp.similarity import compute_similarity
            sem_score = compute_similarity(cv_clean, job_clean)
        except Exception:
            # Fallback : similarité TF-IDF si BERT non disponible
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            vec   = TfidfVectorizer()
            tfidf = vec.fit_transform([cv_clean, job_clean])
            sem_score = float(cos_sim(tfidf[0], tfidf[1])[0][0])

        # Score final avec pondération personnalisée
        from app.config import WEIGHTS
        score = compute_score(
            semantic_similarity=sem_score,
            skills_match_rate=skills_result["match_rate"],
        )

        candidates.append({
            "name":                record.get("id", f"CV_{i+1}"),
            "category":            record.get("category", "—"),
            "final_score":         score["final_score"],
            "percent":             score["percent"],
            "level":               score["level"],
            "label":               score["label"],
            "semantic_similarity": sem_score,
            "skills":              skills_result,
        })

        progress.progress((i + 1) / total, text=f"Analyse {i+1}/{total}...")

    progress.empty()

    # Classement
    ranked = rank_candidates(candidates)[:top_n]
    df     = to_dataframe(ranked)

    st.success(f"✅ Analyse terminée — **{len(cv_records)} CVs** analysés")

    # ── Métriques globales ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Résultats globaux</div>', unsafe_allow_html=True)

    avg_score  = sum(c["percent"] for c in candidates) / len(candidates)
    n_excl     = sum(1 for c in candidates if c["percent"] >= 75)
    n_bon      = sum(1 for c in candidates if 55 <= c["percent"] < 75)
    best_score = max(c["percent"] for c in candidates)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(candidates)}</div><div class="label">CVs analysés</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="value">{round(avg_score)}%</div><div class="label">Score moyen</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="value">{n_excl}</div><div class="label">Excellent match</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="value">{best_score}%</div><div class="label">Meilleur score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tableau de classement ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Classement des candidats</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Détail du meilleur candidat ───────────────────────────────────────────
    st.markdown('<div class="section-title">🥇 Détail — Meilleur candidat</div>', unsafe_allow_html=True)

    best = ranked[0]
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Score final", f"{best['percent']}%")
    with b2:
        st.metric("Similarité BERT", f"{round(best['semantic_similarity']*100)}%")
    with b3:
        matched = best["skills"].get("matched", [])
        job_sk  = best["skills"].get("job_skills", [])
        st.metric("Skills matchées", f"{len(matched)}/{len(job_sk)}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**✅ Compétences présentes :**")
        if matched:
            tags = " ".join([f'<span class="skill-tag">{s}</span>' for s in matched])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.write("—")
    with col_b:
        missing = best["skills"].get("missing", [])
        st.markdown("**❌ Compétences manquantes :**")
        if missing:
            tags = " ".join([f'<span class="skill-tag skill-tag-missing">{s}</span>' for s in missing])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.write("Aucune — profil complet ✅")

    # ── Visualisations ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Visualisations</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        fig, ax = plt.subplots(figsize=(6, 4))
        all_scores = [c["percent"] for c in candidates]
        ax.hist(all_scores, bins=20, color="#3182ce", edgecolor="white", alpha=0.85)
        ax.axvline(x=75, color="#276749", linestyle="--", linewidth=1.5, label="Excellent (75%)")
        ax.axvline(x=55, color="#d69e2e", linestyle="--", linewidth=1.5, label="Bon (55%)")
        ax.set_title("Distribution des scores", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score (%)")
        ax.set_ylabel("Nombre de CVs")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with v2:
        labels  = ["🟢 Excellent", "🟡 Bon", "🟠 Moyen", "🔴 Faible"]
        sizes   = [
            sum(1 for c in candidates if c["percent"] >= 75),
            sum(1 for c in candidates if 55 <= c["percent"] < 75),
            sum(1 for c in candidates if 35 <= c["percent"] < 55),
            sum(1 for c in candidates if c["percent"] < 35),
        ]
        colors  = ["#48bb78", "#ecc94b", "#ed8936", "#fc8181"]
        sizes_f = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]

        if sizes_f:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.pie(
                [s[0] for s in sizes_f],
                labels=[s[1] for s in sizes_f],
                colors=[s[2] for s in sizes_f],
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 9},
            )
            ax2.set_title("Répartition des niveaux", fontsize=12, fontweight="bold")
            st.pyplot(fig2)
            plt.close()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💾 Export</div>', unsafe_allow_html=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger les résultats (CSV)",
        data=csv_data,
        file_name="resume_screening_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#718096; font-size:0.8rem;'>"
    "Resume Screening App · NLP & Semantic Matching · BERT all-MiniLM-L6-v2"
    "</p>",
    unsafe_allow_html=True,
)


def main():
    pass  # Streamlit s'exécute au niveau module, main() est appelé par run.py