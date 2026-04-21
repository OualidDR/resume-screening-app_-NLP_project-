"""
test_preprocessing.py — Tests unitaires pour le preprocessing
Lancer : python -m pytest tests/test_preprocessing.py -v
"""

from app.preprocessing.text_cleaner import clean_text, clean_records


def test_clean_text_removes_urls():
    text = "Visit https://linkedin.com/in/john for more info"
    result = clean_text(text)
    assert "http" not in result
    assert "linkedin" not in result


def test_clean_text_removes_email():
    text = "Contact me at john.doe@gmail.com"
    result = clean_text(text)
    assert "@" not in result


def test_clean_text_lowercase():
    text = "Python Machine Learning NLP"
    result = clean_text(text)
    assert result == result.lower()


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_records_adds_clean_text():
    records = [
        {"id": "1", "category": "HR", "text": "HR Manager with https://example.com"},
        {"id": "2", "category": "IT", "text": "Python developer skilled in ML"},
    ]
    result = clean_records(records)
    assert all("clean_text" in r for r in result)
    assert "http" not in result[0]["clean_text"]


def test_clean_text_normalizes_spaces():
    text = "python   machine    learning"
    result = clean_text(text)
    assert "  " not in result