"""Conftest – add src/ to the Python path for all tests."""

import sys
from pathlib import Path

# Ensure editable installs are not required during test runs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
