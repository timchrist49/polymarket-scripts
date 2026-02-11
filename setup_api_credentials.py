#!/usr/bin/env python3
"""
Generate Polymarket API credentials from private key.

This script derives the L2 API credentials (key, secret, passphrase)
from your L1 private key using Polymarket's official py-clob-client.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from polymarket.config import Settings
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

def setup_api_credentials():
    """Derive and display API credentials."""

    print("=" * 60)
    print("🔐 POLYMARKET API CREDENTIAL SETUP")
    print("=" * 60)
    print()

    settings = Settings()

    if not settings.private_key:
        print("❌ ERROR: No private key found in .env")
        print("   Set POLYMARKET_PRIVATE_KEY in your .env file")
        return False

    print(f"✅ Private key loaded: {settings.private_key[:10]}...")
    print()

    print("🔄 Deriving API credentials from private key...")
    print("   (This uses Polymarket's official derivation method)")
    print()

    try:
        # Create ClobClient with private key
        # The client will automatically derive or create API credentials
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=settings.private_key,
            chain_id=settings.chain_id,
            signature_type=0,  # EOA (Externally Owned Account)
            funder=None
        )

        # Get or derive API credentials
        creds: ApiCreds = client.create_or_derive_api_creds()

        print("=" * 60)
        print("✅ API CREDENTIALS GENERATED!")
        print("=" * 60)
        print()
        print("Add these to your .env file:")
        print()
        print(f"POLYMARKET_API_KEY={creds.api_key}")
        print(f"POLYMARKET_API_SECRET={creds.api_secret}")
        print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
        print()
        print("=" * 60)
        print()
        print("📝 To update .env automatically, run:")
        print()
        print("   sed -i 's/^POLYMARKET_API_KEY=.*/POLYMARKET_API_KEY=" + creds.api_key + "/' .env")
        print("   sed -i 's/^POLYMARKET_API_SECRET=.*/POLYMARKET_API_SECRET=" + creds.api_secret + "/' .env")
        print("   sed -i 's/^POLYMARKET_API_PASSPHRASE=.*/POLYMARKET_API_PASSPHRASE=" + creds.api_passphrase + "/' .env")
        print()

        return True

    except Exception as e:
        print("=" * 60)
        print("❌ CREDENTIAL GENERATION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup_api_credentials()
    sys.exit(0 if success else 1)
