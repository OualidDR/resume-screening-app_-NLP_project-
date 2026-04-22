# Les imports lourds (embeddings, similarity) sont faits à la demande
# pour éviter l'erreur si sentence_transformers n'est pas installé localement
from .skill_extraction import extract_skills, extract_skills_flat, match_skills