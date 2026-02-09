"""
Polymarket authentication handling.

This module manages authentication for the Polymarket CLOB API including
L1 (private key signing) and L2 (API key credentials) authentication.
It supports both Web3 wallets and Gmail/Magic Link custodial accounts.

Classes:
    PolymarketAuth: Handles authentication credential derivation

Constants:
    CLOB_HOST: Production CLOB API endpoint
    CLOB_CHAIN_ID: Polygon chain ID (137)

Example:
    >>> from polymarket.auth import PolymarketAuth
    >>> auth = PolymarketAuth(private_key="0x...")
    >>> credentials = auth.derive_credentials()

Account Types:
    - Web3 Wallets (MetaMask, etc.): signature_type=2
    - Gmail/Magic Link: signature_type=1, requires funder address
"""

from polymarket.config import Settings
from polymarket.exceptions import ConfigError


class AuthManager:
    """
    Manages authentication credentials and mode.

    Supports two modes:
    - read_only: No credentials needed for public market data
    - trading: Requires L1 (private key) and optionally L2 (API credentials)
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

        For L2 operations, User API credentials are automatically derived from
        the private key using create_or_derive_api_creds().

        Returns:
            Dictionary of keyword arguments for ClobClient
        """
        kwargs = {
            "host": self._settings.clob_url,
            "chain_id": self._settings.chain_id,
        }

        if self._mode == "trading":
            kwargs["key"] = self._private_key

            # L2 credentials: Use env vars if provided, otherwise derive from private key
            # For L2 operations, we need: signature_type=2 and optionally funder
            if self.has_api_credentials():
                # Use provided L2 credentials (manual override)
                from py_clob_client.clob_types import ApiCreds
                kwargs["creds"] = ApiCreds(
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                    api_passphrase=self._api_passphrase,
                )
            else:
                # Auto-derive User API credentials from private key
                # This creates/derives the credentials needed for L2 operations
                from py_clob_client.clob_types import ApiCreds
                from py_clob_client.client import ClobClient

                temp_client = ClobClient(
                    host=self._settings.clob_url,
                    chain_id=self._settings.chain_id,
                    key=self._private_key,
                )
                derived = temp_client.create_or_derive_api_creds()
                kwargs["creds"] = derived

            # L2 requires signature_type
            # signature_type=1: Magic/email-based proxy wallet (Gmail accounts)
            # signature_type=2: Deployed Safe proxy wallet (Web3 wallet)
            kwargs["signature_type"] = self._settings.signature_type or 1

            # Add funder if set (optional for some operations)
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
