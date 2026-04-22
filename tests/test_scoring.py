"""
test_scoring.py — Tests unitaires pour le scoring et le ranking
Lancer : python -m pytest tests/test_scoring.py -v
"""

import pandas as pd
from app.scoring.scorer import compute_score, score_cv
from app.scoring.ranking import rank_candidates, to_dataframe, get_top_n


# ── Tests compute_score ───────────────────────────────────────────────────────

def test_compute_score_excellent():
    result = compute_score(semantic_similarity=0.90, skills_match_rate=0.90)
    assert result["percent"] >= 75
    assert result["level"] == "Excellent"
    assert "🟢" in result["emoji"]


def test_compute_score_bon():
    result = compute_score(semantic_similarity=0.65, skills_match_rate=0.55)
    assert 55 <= result["percent"] < 75
    assert result["level"] == "Bon"


def test_compute_score_moyen():
    result = compute_score(semantic_similarity=0.40, skills_match_rate=0.30)
    assert 35 <= result["percent"] < 55
    assert result["level"] == "Moyen"


def test_compute_score_faible():
    result = compute_score(semantic_similarity=0.10, skills_match_rate=0.00)
    assert result["percent"] < 35
    assert result["level"] == "Faible"


def test_compute_score_between_0_and_100():
    result = compute_score(semantic_similarity=0.75, skills_match_rate=0.60)
    assert 0 <= result["percent"] <= 100


def test_compute_score_has_all_keys():
    result = compute_score(semantic_similarity=0.70, skills_match_rate=0.50)
    required_keys = {"final_score", "percent", "level", "emoji", "label",
                     "semantic_similarity", "skills_match_rate"}
    assert required_keys.issubset(set(result.keys()))


def test_compute_score_weights():
    """60% BERT + 40% skills = score attendu."""
    result = compute_score(semantic_similarity=1.0, skills_match_rate=1.0)
    assert result["final_score"] == 1.0


# ── Tests score_cv ────────────────────────────────────────────────────────────

def test_score_cv_returns_skills():
    cv  = "python pandas scikit-learn machine learning nlp bert"
    job = "python machine learning nlp bert data scientist"
    result = score_cv(cv, job, semantic_similarity=0.80)
    assert "skills" in result
    assert "matched" in result["skills"]
    assert "missing" in result["skills"]


def test_score_cv_relevant_higher_than_irrelevant():
    job = "python data scientist machine learning nlp bert"
    cv_relevant   = "python pandas scikit-learn nlp bert machine learning"
    cv_irrelevant = "chef cuisinier restauration gastronomique cuisine"
    score_rel = score_cv(cv_relevant,   job, semantic_similarity=0.85)
    score_irr = score_cv(cv_irrelevant, job, semantic_similarity=0.10)
    assert score_rel["final_score"] > score_irr["final_score"]


# ── Tests rank_candidates ─────────────────────────────────────────────────────

def test_rank_candidates_sorted():
    """Les candidats doivent être triés par score décroissant."""
    candidates = [
        {"name": "cv1.pdf", "final_score": 0.55, "percent": 55, "label": "🟡 Bon", "skills": {}},
        {"name": "cv2.pdf", "final_score": 0.82, "percent": 82, "label": "🟢 Excellent", "skills": {}},
        {"name": "cv3.pdf", "final_score": 0.30, "percent": 30, "label": "🔴 Faible", "skills": {}},
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0]["final_score"] == 0.82
    assert ranked[1]["final_score"] == 0.55
    assert ranked[2]["final_score"] == 0.30


def test_rank_candidates_adds_rank():
    """Le rang doit être ajouté à chaque candidat."""
    candidates = [
        {"name": "cv1.pdf", "final_score": 0.70, "percent": 70, "label": "🟡 Bon", "skills": {}},
        {"name": "cv2.pdf", "final_score": 0.90, "percent": 90, "label": "🟢 Excellent", "skills": {}},
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2


def test_rank_candidates_empty():
    assert rank_candidates([]) == []


def test_get_top_n():
    candidates = [
        {"name": f"cv{i}.pdf", "final_score": 1.0 - i * 0.1,
         "percent": 100 - i * 10, "label": "🟢", "skills": {}}
        for i in range(10)
    ]
    ranked = rank_candidates(candidates)
    top3   = get_top_n(ranked, n=3)
    assert len(top3) == 3
    assert top3[0]["rank"] == 1


# ── Tests to_dataframe ────────────────────────────────────────────────────────

def test_to_dataframe_returns_dataframe():
    candidates = [
        {
            "name": "cv1.pdf", "final_score": 0.80, "percent": 80,
            "label": "🟢 Excellent match — 80%", "level": "Excellent",
            "semantic_similarity": 0.85, "rank": 1,
            "skills": {"matched": ["python", "nlp"], "job_skills": ["python", "nlp"], "missing": []}
        }
    ]
    df = to_dataframe(candidates)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Score" in df.columns
    assert "Candidat" in df.columns


def test_to_dataframe_empty():
    df = to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0