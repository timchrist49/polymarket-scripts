# polymarket/config.py
"""
Polymarket configuration from environment variables.

This module defines the configuration dataclass that reads settings from
environment variables. It supports both read_only and trading modes.

Classes:
    Settings: Configuration container with environment variable loading

Environment Variables:
    POLYMARKET_MODE: "read_only" or "trading"
    POLYMARKET_PRIVATE_KEY: Wallet private key (trading mode)
    POLYMARKET_API_KEY: API key for L2 auth (trading mode)
    POLYMARKET_API_SECRET: API secret for L2 auth (trading mode)
    POLYMARKET_API_PASSPHRASE: API passphrase (trading mode)
    POLYMARKET_FUNDER: Proxy wallet address (Gmail accounts)
    POLYMARKET_SIGNATURE_TYPE: 1 for Magic, 2 for Web3 (default: 1)

Example:
    >>> from polymarket.config import Settings
    >>> settings = Settings()
    >>> if settings.mode == Mode.TRADING:
    ...     print("Trading mode enabled")
"""

import os
from typing import Literal
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings from environment variables."""

    # Runtime mode
    mode: Literal["read_only", "trading"] = field(
        default_factory=lambda: os.getenv("POLYMARKET_MODE", "read_only")
    )

    # L1 Authentication (private key)
    private_key: str | None = field(
        default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY")
    )

    # L2 Authentication (API credentials)
    api_key: str | None = field(
        default_factory=lambda: os.getenv("POLYMARKET_API_KEY")
    )
    api_secret: str | None = field(
        default_factory=lambda: os.getenv("POLYMARKET_API_SECRET")
    )
    api_passphrase: str | None = field(
        default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE")
    )

    # Funder address (proxy wallet that holds funds for Gmail/Magic accounts)
    funder: str | None = field(
        default_factory=lambda: os.getenv("POLYMARKET_FUNDER")
    )

    # Signature type for L2 operations
    # 1 = Magic/email-based proxy wallet (Gmail accounts)
    # 2 = Deployed Safe proxy wallet (Web3 wallets)
    signature_type: int | None = field(
        default_factory=lambda: int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))  # Default to 1 for Gmail
    )

    # Chain configuration
    chain_id: int = field(
        default_factory=lambda: int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
    )

    # API endpoints
    clob_url: str = field(
        default_factory=lambda: os.getenv(
            "POLYMARKET_CLOB_URL", "https://clob.polymarket.com"
        )
    )
    gamma_url: str = field(
        default_factory=lambda: os.getenv(
            "POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"
        )
    )
    data_url: str = field(
        default_factory=lambda: os.getenv(
            "POLYMARKET_DATA_URL", "https://data-polymarket.com"
        )
    )

    # Runtime options
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_json: bool = field(
        default_factory=lambda: os.getenv("LOG_JSON", "false").lower() == "true"
    )

    # === OpenAI Configuration ===
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    # === OpenAI GPT-5-Nano Configuration ===
    openai_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("OPENAI_REASONING_EFFORT", "low")  # low=faster for 15-min markets
    )

    # === Tavily Configuration ===
    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # === BTC Price Service ===
    btc_price_source: str = field(
        default_factory=lambda: os.getenv("BTC_PRICE_SOURCE", "binance")
    )
    btc_price_cache_seconds: int = field(
        default_factory=lambda: int(os.getenv("BTC_PRICE_CACHE_SECONDS", "30"))
    )

    # === Trading Bot Configuration ===
    bot_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("BOT_INTERVAL_SECONDS", "180"))
    )
    bot_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("BOT_CONFIDENCE_THRESHOLD", "0.75"))
    )
    bot_max_position_percent: float = field(
        default_factory=lambda: float(os.getenv("BOT_MAX_POSITION_PERCENT", "0.10"))
    )
    bot_max_exposure_percent: float = field(
        default_factory=lambda: float(os.getenv("BOT_MAX_EXPOSURE_PERCENT", "0.50"))
    )

    # === Stop-Loss Configuration ===
    stop_loss_odds_threshold: float = field(
        default_factory=lambda: float(os.getenv("STOP_LOSS_ODDS_THRESHOLD", "0.40"))
    )
    stop_loss_force_exit_minutes: int = field(
        default_factory=lambda: int(os.getenv("STOP_LOSS_FORCE_EXIT_MINUTES", "5"))
    )

    # === Bot Logging ===
    bot_log_decisions: bool = field(
        default_factory=lambda: os.getenv("BOT_LOG_DECISIONS", "true").lower() == "true"
    )
    bot_log_file: str = field(
        default_factory=lambda: os.getenv("BOT_LOG_FILE", "logs/auto_trade.log")
    )

    def __post_init__(self):
        """Validate settings based on mode."""
        if self.mode == "trading":
            if not self.private_key:
                raise ValueError(
                    "POLYMARKET_PRIVATE_KEY is required for TRADING mode"
                )
            # API credentials are optional - can be derived from private key

    def __repr__(self) -> str:
        """Return masked representation to protect sensitive credentials."""
        # Helper function to mask sensitive values
        def mask(value: str | None) -> str:
            if value is None:
                return "None"
            if len(value) <= 10:
                return "***"
            return f"{value[:6]}...{value[-4:]}"

        return (
            f"Settings("
            f"mode='{self.mode}', "
            f"private_key={mask(self.private_key)}, "
            f"api_key={mask(self.api_key)}, "
            f"api_secret={mask(self.api_secret)}, "
            f"api_passphrase={mask(self.api_passphrase)}, "
            f"funder={self.funder}, "
            f"chain_id={self.chain_id}, "
            f"clob_url='{self.clob_url}', "
            f"gamma_url='{self.gamma_url}', "
            f"data_url='{self.data_url}', "
            f"dry_run={self.dry_run}, "
            f"log_level='{self.log_level}', "
            f"log_json={self.log_json})"
        )


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset settings (mainly for testing)."""
    global _settings
    _settings = None
