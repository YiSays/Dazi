"""Configuration for dazi."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Define project root first
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env from project root
DOTENV_PATH = PROJECT_ROOT / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=True)


# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # for compatible APIs

# --- Paths ---
DATA_DIR = PROJECT_ROOT / ".dazi"
SESSION_DIR = DATA_DIR / "sessions"
