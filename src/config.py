
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from src/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ── Google Gemini ─────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-3.6-flash"   # updated model name

# ── Embeddings (free, local) ──────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Document chunking ─────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────
TOP_K_RETRIEVAL = 5

# ── FAISS index storage ───────────────────────
FAISS_INDEX_PATH = str(_project_root / "data" / "faiss_index")

print(f"🔑 API key loaded: {'✅ Yes' if GOOGLE_API_KEY else '❌ Missing'}")