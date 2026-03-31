from dotenv import load_dotenv
import os

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

MODEL: str = "claude-sonnet-4-20250514"
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL: str = "gemini-2.5-flash-image"
TEMPERATURE: float = 0
MAX_TOKENS: int = 4096

ML_APP_ID: str = os.getenv("ML_APP_ID", "")
ML_SECRET_KEY: str = os.getenv("ML_SECRET_KEY", "")
ML_REDIRECT_URI: str = os.getenv("ML_REDIRECT_URI", "https://localhost")
