"""
logger.py — Configuration centralisée du logging avec loguru
Usage dans n'importe quel module :
    from app.utils.logger import logger
    logger.info("message")
"""

import sys
from loguru import logger
from app.config import LOG_LEVEL

# Supprimer le handler par défaut et en créer un propre
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    colorize=True,
)

# Export unique : tout le projet importe 'logger' depuis ici
__all__ = ["logger"]