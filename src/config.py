from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_media"
ASSETS_DIR = BASE_DIR / "assets"
HISTORY_PATH = DATA_DIR / "run_history.csv"
SUPPORTED_IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
SUPPORTED_DOC_TYPES = [".pdf"]
