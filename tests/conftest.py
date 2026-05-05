import sys
from pathlib import Path

# Make sure ScratchRAG root is on the path so "import minrag" and "import api" work
sys.path.insert(0, str(Path(__file__).parent.parent))
