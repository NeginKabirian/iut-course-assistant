import os
from dotenv import load_dotenv

load_dotenv()

for k in ("DATABASE_URL", "BASE_URL", "API_KEY"):
    if not os.environ.get(k):
        raise RuntimeError(f"Missing environment variable: {k}")

DATABASE_URL = os.environ.get("DATABASE_URL")
BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

COLLECTION_NAME = "iut_courses"
DEFAULT_TEMPERATURE = 0.1
MAX_HISTORY = 12
