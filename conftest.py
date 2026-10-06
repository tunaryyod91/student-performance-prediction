import sys
from pathlib import Path

# Assure que le dossier racine est dans sys.path pour pytest
sys.path.insert(0, str(Path(__file__).parent))
