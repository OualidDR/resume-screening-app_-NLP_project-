"""
test_skills.py — Tests unitaires pour l'extraction de compétences
Lancer : python -m pytest tests/test_skills.py -v
"""

from app.nlp.skill_extraction import (
    extract_skills,
    extract_skills_flat,
    match_skills,
    load_skills_db,
)


# ── Tests load_skills_db ──────────────────────────────────────────────────────

def test_load_skills_db_returns_dict():
    db = load_skills_db()
    assert isinstance(db, dict)
    assert len(db) > 0


def test_load_skills_db_has_categories():
    db = load_skills_db()
    expected = {"langages", "frameworks_ml", "nlp", "data"}
    assert expected.issubset(set(db.keys()))


# ── Tests extract_skills ──────────────────────────────────────────────────────

def test_extract_skills_finds_python():
    result = extract_skills("experienced python developer with sql skills")
    assert "langages" in result
    assert "python" in result["langages"]


def test_extract_skills_finds_nlp():
    result = extract_skills("nlp engineer with bert and word2vec experience")
    assert "nlp" in result
    assert "bert" in result["nlp"]


def test_extract_skills_empty_text():
    result = extract_skills("")
    assert result == {}


def test_extract_skills_no_match():
    result = extract_skills("chef cuisinier gastronomique restaurant étoilé")
    total = sum(len(v) for v in result.values())
    assert total <= 1  # "r" peut matcher mais rien d'autre



def test_extract_skills_multiple_categories():
    text = "python pandas bert docker postgresql"
    result = extract_skills(text)
    assert len(result) >= 3   # langages + data + cloud_devops ou nlp


# ── Tests extract_skills_flat ─────────────────────────────────────────────────

def test_extract_skills_flat_returns_list():
    result = extract_skills_flat("python sql pandas bert")
    assert isinstance(result, list)
    assert "python" in result


def test_extract_skills_flat_no_duplicates():
    result = extract_skills_flat("python python python")
    assert result.count("python") == 1


# ── Tests match_skills ────────────────────────────────────────────────────────

def test_match_skills_perfect_match():
    cv  = "python pandas scikit-learn nlp bert sql"
    job = "python pandas nlp bert"
    result = match_skills(cv, job)
    assert result["match_rate"] == 1.0
    assert len(result["missing"]) == 0


def test_match_skills_partial_match():
    cv  = "python pandas"
    job = "python pandas scikit-learn bert nlp"
    result = match_skills(cv, job)
    assert 0.0 < result["match_rate"] < 1.0
    assert len(result["missing"]) > 0


def test_match_skills_no_match():
    cv  = "chef cuisinier restauration"
    job = "python machine learning nlp bert"
    result = match_skills(cv, job)
    assert result["match_rate"] <= 0.25  # au pire "r" matche
    assert "python" not in result["matched"]


def test_match_skills_returns_all_keys():
    result = match_skills("python sql", "python nlp bert")
    keys = {"cv_skills", "job_skills", "matched", "missing", "match_rate"}
    assert keys.issubset(set(result.keys()))


def test_match_skills_empty_job():
    result = match_skills("python pandas nlp", "")
    assert result["match_rate"] == 0.0


def test_match_skills_matched_subset_of_job():
    """Les compétences matchées doivent être un sous-ensemble des skills du poste."""
    cv  = "python pandas docker"
    job = "python nlp bert"
    result = match_skills(cv, job)
    for skill in result["matched"]:
        assert skill in result["job_skills"]