"""Streamlit entrypoint: `streamlit run app.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vehicle_recognition.app.main import main  # noqa: E402

main()
