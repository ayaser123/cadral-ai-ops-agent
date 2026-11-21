"""All configuration comes from environment variables (never hard-coded)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def env(name, default=""):
    return os.environ.get(name, default)


GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_MODEL_MAIN = env("GROQ_MODEL_MAIN", "openai/gpt-oss-120b")
GROQ_MODEL_FAST = env("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
FT_BASE_URL = env("FT_BASE_URL")
FT_MODEL = env("FT_MODEL", "cadral-router-1b")
SECRET_KEY = env("SECRET_KEY", "dev-secret-change-me")
DB_PATH = Path(env("DB_PATH", str(ROOT / "data" / "cadral.db")))
DOCS_DIR = ROOT / "data" / "docs"
AUTO_REFUND_LIMIT = float(env("AUTO_REFUND_LIMIT_GBP", "100"))
SLACK_WEBHOOK_URL = env("SLACK_WEBHOOK_URL")
SMTP_HOST, SMTP_PORT = env("SMTP_HOST"), int(env("SMTP_PORT", "587") or 587)
SMTP_USER, SMTP_PASSWORD = env("SMTP_USER"), env("SMTP_PASSWORD")
SMTP_FROM = env("SMTP_FROM", "support@cadral.test")
MANAGER_EMAIL = env("MANAGER_EMAIL")
IMAP_HOST, IMAP_USER, IMAP_PASSWORD = env("IMAP_HOST"), env("IMAP_USER"), env("IMAP_PASSWORD")
TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN")
REPORT_HOUR_UTC = int(env("REPORT_HOUR_UTC", "7") or 7)
ADMIN_EMAIL, ADMIN_PASSWORD = env("ADMIN_EMAIL", "admin@cadral.test"), env("ADMIN_PASSWORD", "ChangeMe-Admin1")
AGENT_EMAIL, AGENT_PASSWORD = env("AGENT_EMAIL", "agent@cadral.test"), env("AGENT_PASSWORD", "ChangeMe-Agent1")

# USD per 1M tokens (list prices; Groq free tier is $0 but we report the "would-cost" figure)
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "cadral-router-1b": (0.0, 0.0),
    "offline-heuristic": (0.0, 0.0),
}
