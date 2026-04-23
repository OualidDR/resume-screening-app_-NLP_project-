"""
main.py — Interface Streamlit Premium — Dark Blue & Gold
Resume Screening App · NLP & Semantic Matching
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

from app.preprocessing.extractor import extract_text, load_dataset_csv
from app.preprocessing.text_cleaner import clean_text
from app.nlp.skill_extraction import match_skills, load_skills_db, extract_skills_flat
from app.scoring.scorer import compute_score
from app.scoring.ranking import rank_candidates, to_dataframe

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ · Screening Intelligent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
/* Cacher SEULEMENT le bouton Deploy */
[data-testid="stToolbarActionButtonIcon"] {
    display: none !important;
}
button[kind="deployButton"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
# ── CSS Premium Dark Blue & Gold ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070b14;
    color: #e2e8f0;
}
.main .block-container {
    background: #070b14;
    padding: 2rem 2.5rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1526 100%);
    border-right: 1px solid rgba(212, 175, 55, 0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #d4af37 !important; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1f3d 50%, #0a1628 100%);
    border: 1px solid rgba(212, 175, 55, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(212,175,55,0.06) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #d4af37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub { color: #8899bb; font-size: 1rem; margin: 0.6rem 0 0 0; font-weight: 300; }
.hero-badge {
    display: inline-block;
    background: rgba(212,175,55,0.1);
    border: 1px solid rgba(212,175,55,0.4);
    color: #d4af37;
    font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-right: 6px; letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-card {
    background: linear-gradient(135deg, #0d1526 0%, #0f1a2e 100%);
    border: 1px solid rgba(212,175,55,0.15);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4af37, transparent);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-label {
    color: #667a99; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-top: 0.4rem; font-weight: 500;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem; font-weight: 700;
    color: #d4af37; text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 2rem 0 1rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(212,175,55,0.3), transparent);
    margin-left: 0.5rem;
}
.progress-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 50px; height: 6px; overflow: hidden; margin: 6px 0;
}
.progress-fill {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #1a3a6e, #d4af37);
}
.badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; }
.badge-excellent { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-bon       { background: rgba(212,175,55,0.12);  color: #d4af37; border: 1px solid rgba(212,175,55,0.3); }
.badge-moyen     { background: rgba(251,146,60,0.12);  color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
.badge-faible    { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.skill-tag         { display:inline-block; background:rgba(212,175,55,0.08);   border:1px solid rgba(212,175,55,0.25);   color:#d4af37; border-radius:6px; padding:3px 10px; font-size:0.75rem; margin:2px; }
.skill-tag-matched { background:rgba(52,211,153,0.08);  border-color:rgba(52,211,153,0.25);  color:#34d399; border-radius:6px; padding:3px 10px; font-size:0.75rem; margin:2px; display:inline-block; }
.skill-tag-missing { background:rgba(248,113,113,0.08); border-color:rgba(248,113,113,0.25); color:#f87171; border-radius:6px; padding:3px 10px; font-size:0.75rem; margin:2px; display:inline-block; }
.candidate-card {
    background: linear-gradient(135deg, #0d1526, #0f1a2e);
    border: 1px solid rgba(212,175,55,0.1);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.3), transparent);
    margin: 1.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3a6e, #0f2450) !important;
    color: #d4af37 !important;
    border: 1px solid rgba(212,175,55,0.4) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #d4af37, #b8930a) !important;
    color: #070b14 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="margin-bottom:0.8rem;">
        <span class="hero-badge">NLP</span>
        <span class="hero-badge">BERT</span>
        <span class="hero-badge">AI Screening</span>
    </div>
    <h1 class="hero-title">ResumeIQ</h1>
    <p class="hero-sub">Plateforme intelligente de présélection · Semantic Matching · Scoring automatique</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Configuration")
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    mode   = st.radio("Source des CVs", ["📊 Dataset CSV", "📁 Upload PDF/DOCX"])
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 🎛 Scoring")
    w_bert   = st.slider("Poids BERT", 0.0, 1.0, 0.6, 0.05)
    w_skills = round(1.0 - w_bert, 2)
    st.markdown(f"<div style='font-size:0.8rem;color:#667a99;'>BERT <b style='color:#d4af37'>{int(w_bert*100)}%</b> · Skills <b style='color:#d4af37'>{int(w_skills*100)}%</b></div>", unsafe_allow_html=True)
    top_n  = st.slider("Top N candidats", 5, 50, 10)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📋 Pipeline")
    for i, s in enumerate(["Upload CVs","Offre d'emploi","Preprocessing","BERT Embeddings","Scoring","Classement"], 1):
        st.markdown(f"<div style='font-size:0.82rem;padding:3px 0;color:#8899bb;'><span style='color:#d4af37;font-weight:700;'>{i}.</span> {s}</div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
cv_records = []
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-title">📂 Source des CVs</div>', unsafe_allow_html=True)
    if "CSV" in mode:
        uploaded_csv = st.file_uploader("Dataset CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_csv.read())
                tmp_path = tmp.name
            records    = load_dataset_csv(tmp_path)
            os.unlink(tmp_path)
            max_cvs    = st.slider("Nombre de CVs", 10, min(500, len(records)), 100)
            cv_records = records[:max_cvs]
            st.markdown(f"<div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);border-radius:10px;padding:0.8rem 1rem;'><span style='color:#34d399;font-weight:600;'>✓</span> <b style='color:#e2e8f0;'>{len(cv_records)}</b> <span style='color:#a0aec0;'>CVs chargés</span></div>", unsafe_allow_html=True)
            if st.checkbox("👁 Aperçu"):
                df_p = pd.DataFrame(cv_records[:5])[["id","category","text"]]
                df_p["text"] = df_p["text"].str[:80] + "..."
                st.dataframe(df_p, use_container_width=True, hide_index=True)
    else:
        uploaded_files = st.file_uploader("CVs PDF/DOCX", type=["pdf","docx"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            import tempfile, os
            for f in uploaded_files:
                ext = f.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(f.read()); tmp_path = tmp.name
                cv_records.append({"id": f.name, "category": "Upload", "text": extract_text(tmp_path)})
                os.unlink(tmp_path)
            st.success(f"✓ {len(cv_records)} CVs uploadés")

with col_right:
    st.markdown('<div class="section-title">✍️ Offre d\'emploi</div>', unsafe_allow_html=True)
    examples = {
        "— Exemple —": "",
        "🔬 Data Scientist":    "We are looking for a Data Scientist with Python, machine learning and NLP. Required: pandas, scikit-learn, PyTorch, SQL, BERT, transformers.",
        "💰 Finance Analyst":   "Financial Analyst with Excel, VBA, Bloomberg. Financial modeling, risk management. Python appreciated.",
        "💻 Software Engineer": "Backend engineer Python or Java. Docker, Kubernetes, PostgreSQL, REST APIs, Git, CI/CD.",
        "👥 HR Manager":        "HR Manager with recruitment and onboarding experience. Communication, leadership, team management.",
    }
    selected = st.selectbox("Exemple", list(examples.keys()), label_visibility="collapsed")
    job_text = st.text_area("Offre", value=examples[selected], height=160, placeholder="Description du poste...", label_visibility="collapsed")
    if job_text.strip():
        skills_db  = load_skills_db()
        job_skills = extract_skills_flat(job_text, skills_db)
        if job_skills:
            st.markdown("<div style='font-size:0.75rem;color:#667a99;margin:0.5rem 0 0.3rem;'>COMPÉTENCES DÉTECTÉES</div>", unsafe_allow_html=True)
            st.markdown("".join([f'<span class="skill-tag">{s}</span>' for s in job_skills]), unsafe_allow_html=True)

# ── Bouton ────────────────────────────────────────────────────────────────────
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
can_run = len(cv_records) > 0 and len(job_text.strip()) > 0
btn     = st.button("⚡  LANCER L'ANALYSE", disabled=not can_run, use_container_width=True, type="primary")
if not can_run:
    st.markdown("<div style='text-align:center;color:#334455;font-size:0.85rem;padding:0.5rem;'>Upload des CVs et renseigne une offre pour continuer</div>", unsafe_allow_html=True)

# ── Analyse ───────────────────────────────────────────────────────────────────
if btn and can_run:
    skills_db  = load_skills_db()
    job_clean  = clean_text(job_text)
    candidates = []
    prog       = st.progress(0)
    prog_txt   = st.empty()
    total      = len(cv_records)

    for i, record in enumerate(cv_records):
        cv_clean      = clean_text(record["text"])
        skills_result = match_skills(cv_clean, job_clean, skills_db)
        try:
            from app.nlp.similarity import compute_similarity
            sem_score = compute_similarity(cv_clean, job_clean)
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cs
            v = TfidfVectorizer()
            m = v.fit_transform([cv_clean, job_clean])
            sem_score = float(cs(m[0], m[1])[0][0])

        score = compute_score(semantic_similarity=sem_score, skills_match_rate=skills_result["match_rate"])
        candidates.append({
            "name": record.get("id", f"CV_{i+1}"),
            "category": record.get("category", "—"),
            "final_score": score["final_score"],
            "percent": score["percent"],
            "level": score["level"],
            "label": score["label"],
            "semantic_similarity": sem_score,
            "skills": skills_result,
        })
        pct = (i + 1) / total
        prog.progress(pct)
        prog_txt.markdown(f"<div style='text-align:center;color:#667a99;font-size:0.8rem;'>Analyse · {i+1}/{total} · {int(pct*100)}%</div>", unsafe_allow_html=True)

    prog.empty(); prog_txt.empty()
    ranked = rank_candidates(candidates)[:top_n]

    # ── Métriques ──
    st.markdown('<div class="section-title">📊 Résultats globaux</div>', unsafe_allow_html=True)
    avg   = round(sum(c["percent"] for c in candidates) / len(candidates))
    best  = max(c["percent"] for c in candidates)
    n_exc = sum(1 for c in candidates if c["percent"] >= 75)

    m1,m2,m3,m4 = st.columns(4)
    def mc(v, l):
        return f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{l}</div></div>'
    with m1: st.markdown(mc(len(candidates), "CVs analysés"), unsafe_allow_html=True)
    with m2: st.markdown(mc(f"{avg}%", "Score moyen"), unsafe_allow_html=True)
    with m3: st.markdown(mc(n_exc, "Excellent match"), unsafe_allow_html=True)
    with m4: st.markdown(mc(f"{best}%", "Meilleur score"), unsafe_allow_html=True)

    # ── Graphiques Plotly ──
    st.markdown('<div class="section-title">📈 Visualisations</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        scores = [c["percent"] for c in candidates]
        fig1   = go.Figure(go.Histogram(
            x=scores, nbinsx=20,
            marker=dict(color=scores, colorscale=[[0,"#1a3a6e"],[0.5,"#d4af37"],[1,"#f0d080"]], line=dict(color="#070b14",width=1)),
            hovertemplate="Score: %{x}%<br>Count: %{y}<extra></extra>",
        ))
        fig1.add_vline(x=75, line_dash="dash", line_color="#34d399", annotation_text="Excellent", annotation_font_color="#34d399")
        fig1.add_vline(x=55, line_dash="dash", line_color="#d4af37", annotation_text="Bon",       annotation_font_color="#d4af37")
        fig1.update_layout(title=dict(text="Distribution des scores", font=dict(color="#e2e8f0",size=13)),
            paper_bgcolor="#0d1526", plot_bgcolor="#0d1526", font=dict(color="#8899bb"),
            xaxis=dict(gridcolor="#1a2a40"), yaxis=dict(gridcolor="#1a2a40"),
            margin=dict(l=10,r=10,t=40,b=10), height=280)
        st.plotly_chart(fig1, use_container_width=True)

    with g2:
        n_bon  = sum(1 for c in candidates if 55 <= c["percent"] < 75)
        n_moy  = sum(1 for c in candidates if 35 <= c["percent"] < 55)
        n_fai  = sum(1 for c in candidates if c["percent"] < 35)
        data   = [(v,l,c) for v,l,c in zip([n_exc,n_bon,n_moy,n_fai],["Excellent","Bon","Moyen","Faible"],["#34d399","#d4af37","#fb923c","#f87171"]) if v>0]
        fig2   = go.Figure(go.Pie(
            values=[d[0] for d in data], labels=[d[1] for d in data],
            marker=dict(colors=[d[2] for d in data], line=dict(color="#070b14",width=2)),
            hole=0.55, textfont=dict(color="#e2e8f0",size=11),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig2.update_layout(title=dict(text="Répartition des niveaux", font=dict(color="#e2e8f0",size=13)),
            paper_bgcolor="#0d1526", font=dict(color="#8899bb"),
            legend=dict(font=dict(color="#8899bb"), bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text=f"<b>{len(candidates)}</b><br>CVs", font=dict(color="#d4af37",size=13), showarrow=False)],
            margin=dict(l=10,r=10,t=40,b=10), height=280)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Classement ──
    st.markdown('<div class="section-title">🏆 Classement des candidats</div>', unsafe_allow_html=True)
    for c in ranked:
        pct   = c["percent"]
        bcls  = "badge-excellent" if pct>=75 else "badge-bon" if pct>=55 else "badge-moyen" if pct>=35 else "badge-faible"
        rkls  = "top" if c["rank"]<=3 else ""
        matched = c["skills"].get("matched",[])
        job_sk  = c["skills"].get("job_skills",[])
        st.markdown(f"""
        <div class="candidate-card">
          <div style="display:flex;align-items:center;gap:1rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{'#d4af37' if c['rank']<=3 else 'rgba(212,175,55,0.3)'};min-width:40px;">#{c['rank']}</div>
            <div style="flex:1;">
              <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:4px;">
                <span style="font-weight:600;color:#e2e8f0;font-size:0.88rem;">{c['name']}</span>
                <span class="badge {bcls}">{c['level']}</span>
                <span style="color:#445566;font-size:0.75rem;">{c.get('category','—')}</span>
              </div>
              <div style="display:flex;gap:1.5rem;margin-bottom:5px;">
                <span style="font-size:0.78rem;color:#667a99;">Score: <b style="color:#d4af37">{pct}%</b></span>
                <span style="font-size:0.78rem;color:#667a99;">BERT: <b style="color:#8899bb">{round(c['semantic_similarity']*100)}%</b></span>
                <span style="font-size:0.78rem;color:#667a99;">Skills: <b style="color:#8899bb">{len(matched)}/{len(job_sk)}</b></span>
              </div>
              <div class="progress-wrap"><div class="progress-fill" style="width:{pct}%"></div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Profil détaillé ──
    st.markdown('<div class="section-title">🥇 Profil détaillé — Meilleur candidat</div>', unsafe_allow_html=True)
    best_c  = ranked[0]
    matched = best_c["skills"].get("matched",[])
    missing = best_c["skills"].get("missing",[])
    job_sk  = best_c["skills"].get("job_skills",[])

    p1, p2, p3 = st.columns(3)
    with p1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best_c["percent"],
            number=dict(suffix="%", font=dict(color="#d4af37",size=32)),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#667a99", tickfont=dict(color="#667a99")),
                bar=dict(color="#d4af37"),
                bgcolor="#0d1526", bordercolor="#1a2a40",
                steps=[dict(range=[0,35],color="#1a2030"),dict(range=[35,55],color="#1a2535"),dict(range=[55,75],color="#1a2a3a"),dict(range=[75,100],color="#1a3040")],
                threshold=dict(line=dict(color="#34d399",width=2), value=75),
            ),
            title=dict(text="Score final", font=dict(color="#8899bb",size=12)),
        ))
        fig_g.update_layout(paper_bgcolor="#0d1526", font=dict(color="#e2e8f0"), height=210, margin=dict(l=20,r=20,t=30,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with p2:
        st.markdown("<div style='font-size:0.8rem;color:#34d399;font-weight:600;margin-bottom:8px;'>✅ COMPÉTENCES PRÉSENTES</div>", unsafe_allow_html=True)
        if matched:
            st.markdown("".join([f'<span class="skill-tag-matched">{s}</span>' for s in matched]), unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#445566;font-size:0.85rem;'>Aucune détectée</span>", unsafe_allow_html=True)

    with p3:
        st.markdown("<div style='font-size:0.8rem;color:#f87171;font-weight:600;margin-bottom:8px;'>❌ COMPÉTENCES MANQUANTES</div>", unsafe_allow_html=True)
        if missing:
            st.markdown("".join([f'<span class="skill-tag-missing">{s}</span>' for s in missing]), unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#34d399;font-size:0.85rem;'>✓ Profil complet</span>", unsafe_allow_html=True)

    # ── Export ──
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    csv_data = to_dataframe(ranked).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Télécharger les résultats (CSV)", data=csv_data, file_name="resumeiq_results.csv", mime="text/csv", use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem 0;">
    <div class="gold-divider"></div>
    <span style="color:#1a2a40;font-size:0.72rem;letter-spacing:0.1em;">
        RESUMEIQ · NLP & SEMANTIC MATCHING · BERT all-MiniLM-L6-v2
    </span>
</div>
""", unsafe_allow_html=True)

def main():
    pass