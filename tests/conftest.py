"""
conftest.py — Configuration pytest partagée entre tous les tests
Fixtures réutilisables dans tous les fichiers de test.
"""

import pytest


@pytest.fixture
def sample_cv_text():
    return (
        "Experienced data scientist with 5 years in python machine learning. "
        "Skills: pandas, numpy, scikit-learn, bert, nlp, sql, docker, git. "
        "Experience with tensorflow pytorch and transformers. "
        "Strong communication and teamwork skills."
    )


@pytest.fixture
def sample_job_text():
    return (
        "We are looking for a data scientist with python and machine learning expertise. "
        "Required: pandas, scikit-learn, nlp, bert, sql. "
        "Docker and git knowledge appreciated."
    )


@pytest.fixture
def sample_candidates():
    return [
        {
            "name": "alice_cv.pdf",
            "final_score": 0.85,
            "percent": 85,
            "level": "Excellent",
            "label": "🟢 Excellent match — 85%",
            "semantic_similarity": 0.88,
            "rank": 1,
            "skills": {
                "matched": ["python", "nlp", "bert", "pandas"],
                "job_skills": ["python", "nlp", "bert", "pandas"],
                "missing": [],
                "match_rate": 1.0,
            },
        },
        {
            "name": "bob_cv.pdf",
            "final_score": 0.62,
            "percent": 62,
            "level": "Bon",
            "label": "🟡 Bon match — 62%",
            "semantic_similarity": 0.65,
            "rank": 2,
            "skills": {
                "matched": ["python", "pandas"],
                "job_skills": ["python", "nlp", "bert", "pandas"],
                "missing": ["nlp", "bert"],
                "match_rate": 0.5,
            },
        },
        {
            "name": "charlie_cv.pdf",
            "final_score": 0.28,
            "percent": 28,
            "level": "Faible",
            "label": "🔴 Faible match — 28%",
            "semantic_similarity": 0.20,
            "rank": 3,
            "skills": {
                "matched": [],
                "job_skills": ["python", "nlp", "bert", "pandas"],
                "missing": ["python", "nlp", "bert", "pandas"],
                "match_rate": 0.0,
            },
        },
    ]