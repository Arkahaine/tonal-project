# conftest.py
import sys
from pathlib import Path

# Ajouter le répertoire du projet au PYTHONPATH
project_root = Path(__file__).resolve().parent / "src"
sys.path.append(str(project_root))
