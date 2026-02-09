"""Authentication and credential management for Polymarket API."""

from polymarket.config import Settings
from polymarket.exceptions import ConfigError


class AuthManager:
    """
    Manages authentication credentials and mode.

    Supports two modes:
    - READ_ONLY: No credentials needed for public market data
    - TRADING: Requires L1 (private key) and optionally L2 (API credentials)
    """

    def __init__(self, settings: Settings):
        """
        Initialize auth manager from settings.

        Args:
            settings: Application settings

        Raises:
            ConfigError: If TRADING mode but missing required credentials
        """
        self._settings = settings
        self._mode = settings.mode

        # Validate credentials based on mode
        if self._mode == "trading":
            if not settings.private_key:
                raise ConfigError(
                    "POLYMARKET_PRIVATE_KEY is required for TRADING mode"
                )
            self._private_key = settings.private_key
            self._api_key = settings.api_key
            self._api_secret = settings.api_secret
            self._api_passphrase = settings.api_passphrase
            self._funder = settings.funder
        else:
            self._private_key = None
            self._api_key = None
            self._api_secret = None
            self._api_passphrase = None
            self._funder = None

    @property
    def mode(self) -> str:
        """Get current auth mode."""
        return self._mode

    @property
    def private_key(self) -> str | None:
        """Get private key (TRADING mode only)."""
        return self._private_key

    @property
    def api_key(self) -> str | None:
        """Get API key for L2 auth."""
        return self._api_key

    @property
    def api_secret(self) -> str | None:
        """Get API secret for L2 auth."""
        return self._api_secret

    @property
    def api_passphrase(self) -> str | None:
        """Get API passphrase for L2 auth."""
        return self._api_passphrase

    @property
    def funder(self) -> str | None:
        """Get funder address."""
        return self._funder

    def requires_private_key(self) -> bool:
        """Check if current mode requires private key."""
        return self._mode == "trading"

    def has_api_credentials(self) -> bool:
        """Check if L2 API credentials are configured."""
        return bool(
            self._api_key
            and self._api_secret
            and self._api_passphrase
        )

    def get_masked_key(self) -> str | None:
        """
        Get masked private key for safe logging.

        Returns:
            Masked key like "0xaaaa...aaaa" or None if not set
        """
        if not self._private_key:
            return None
        if len(self._private_key) < 10:
            return "0x****"
        return f"{self._private_key[:6]}...{self._private_key[-4:]}"

    def get_clob_client_kwargs(self) -> dict:
        """
        Get kwargs for py-clob-client.ClobClient initialization.

        Returns:
            Dictionary of keyword arguments for ClobClient
        """
        kwargs = {
            "host": self._settings.clob_url,
            "chain_id": self._settings.chain_id,
        }

        if self._mode == "trading":
            kwargs["key"] = self._private_key

            # Add L2 credentials if available
            if self.has_api_credentials():
                kwargs["creds"] = {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "passphrase": self._api_passphrase,
                }

            # Add funder if set
            if self._funder:
                kwargs["funder"] = self._funder

        return kwargs


# Global auth manager instance
_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance (singleton)."""
    global _auth_manager
    if _auth_manager is None:
        from polymarket.config import get_settings
        _auth_manager = AuthManager(get_settings())
    return _auth_manager


def reset_auth_manager():
    """Reset auth manager (mainly for testing)."""
    global _auth_manager
    _auth_manager = None
