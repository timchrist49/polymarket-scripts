"""Config for AI Analysis Upgrade service — all values from environment."""
import os

# Production DB (read ai_analysis_log, write ai_analysis_v2)
PRODUCTION_DB_PATH = os.getenv(
    "POLYMARKET_DB_PATH",
    "/root/polymarket-scripts/data/performance.db"
)

# OpenAI (same key as production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")

# Telegram (same bot + chat as production)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# CoinGecko PRO
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")

# Polling intervals
POLL_INTERVAL_SECONDS = int(os.getenv("AI_UPGRADE_POLL_SECONDS", "30"))

# Kraken OHLC
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

# Polymarket CLOB REST (public, no auth)
CLOB_REST_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
