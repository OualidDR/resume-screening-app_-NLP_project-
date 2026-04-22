"""
run.py — Point d'entrée Streamlit
Usage : streamlit run run.py
"""
import sys
import os

# Ajouter le projet au path
sys.path.insert(0, os.path.dirname(__file__))

# Importer et exécuter main directement
from app.main import *