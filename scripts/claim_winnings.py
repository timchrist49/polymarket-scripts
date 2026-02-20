#!/usr/bin/env python3
"""
Claim Polymarket winnings by redeeming conditional tokens via proxy wallet.

Uses Polymarket's Builder Relayer for gasless transactions — no MATIC required.

Usage:
    python3 scripts/claim_winnings.py [--dry-run] [--all]

Options:
    --dry-run   Preview positions without sending transactions
    --all       Also redeem losing positions (worth $0, just clears them)

Requirements:
    - POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER in .env
    - POLYMARKET_BUILDER_KEY, POLYMARKET_BUILDER_SECRET,
      POLYMARKET_BUILDER_PASSPHRASE, POLYMARKET_RELAYER_URL in .env

How it works:
    1. Queries data-api.polymarket.com for open positions in proxy wallet
    2. Filters positions where curPrice > 0.5 (winning tokens worth ~$1 each)
    3. Gets a relay nonce + relay address from the relayer
    4. Signs a GSN v1-style struct hash authorising the proxy() call
    5. POSTs to /submit — Polymarket's relayer executes on-chain (gasless)
"""

import os
import sys
import json
import time

import requests
from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address
from eth_account import Account
from eth_account.messages import encode_defunct
from dotenv import load_dotenv

from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

# ─── Constants ───────────────────────────────────────────────────────────────

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CHAIN_ID = 137  # Polygon Mainnet

# Proxy contract config for Polygon — from Polymarket TypeScript SDK
PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
RELAY_HUB = "0xD216153c06E857cD7f72665E0aF1d7D82172F494"

# Gas limit for the GSN relay call.
# Must be small enough that the relay can provide gasleft() >= gasLimit + reserve.
# Polymarket's relay caps outer transaction gas; 10_000_000 causes "Not enough gasleft()".
# 300_000 is sufficient for CTF redeemPositions (~100-150k actual gas) with headroom.
DEFAULT_GAS_LIMIT = 300_000

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

# ─── ABI Encoding ────────────────────────────────────────────────────────────

# Function selectors
REDEEM_SELECTOR = keccak(b"redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
PROXY_SELECTOR = keccak(b"proxy((uint8,address,uint256,bytes)[])")[:4]


def encode_redeem_calldata(condition_id_hex: str) -> bytes:
    """Encode redeemPositions(USDC, bytes32(0), conditionId, [1,2]) calldata."""
    condition_id = bytes.fromhex(condition_id_hex.replace("0x", ""))
    args = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [USDC_ADDRESS, b"\x00" * 32, condition_id, [1, 2]],
    )
    return REDEEM_SELECTOR + args


def encode_proxy_calldata(inner_calldata: bytes) -> bytes:
    """
    Encode proxy([(typeCode=1, to=CTF, value=0, data=calldata)]) calldata.

    typeCode=1 = standard Call (not delegatecall).
    This calldata goes to the ProxyWalletFactory, which routes it to the
    user's proxy wallet.
    """
    args = abi_encode(
        ["(uint8,address,uint256,bytes)[]"],
        [[(1, CTF_ADDRESS, 0, inner_calldata)]],
    )
    return PROXY_SELECTOR + args


# ─── PROXY Transaction Signing ───────────────────────────────────────────────

def build_proxy_struct_hash(
    from_addr: str,
    proxy_factory: str,
    data: bytes,
    nonce: str,
    relay_addr: str,
    gas_limit: int = DEFAULT_GAS_LIMIT,
) -> bytes:
    """
    Create the GSN v1-style struct hash for a PROXY transaction.

    Replicates TypeScript createStructHash() from builder-relayer-client:
      keccak256(concat([
        "rlx:",           // relay hub prefix (4 bytes)
        from,             // EOA address (20 bytes)
        to,               // ProxyFactory address (20 bytes)
        data,             // calldata (variable)
        txFee uint256,    // relayer fee = 0 (32 bytes)
        gasPrice uint256, // gas price = 0 (32 bytes)
        gasLimit uint256, // gas limit (32 bytes)
        nonce uint256,    // nonce from relay-payload (32 bytes)
        relayHub,         // RelayHub address (20 bytes)
        relay,            // relay address from relay-payload (20 bytes)
      ]))
    """
    prefix = b"rlx:"
    from_bytes = bytes.fromhex(from_addr.replace("0x", "").lower())
    to_bytes = bytes.fromhex(proxy_factory.replace("0x", "").lower())
    tx_fee_bytes = (0).to_bytes(32, "big")
    gas_price_bytes = (0).to_bytes(32, "big")
    gas_limit_bytes = gas_limit.to_bytes(32, "big")
    nonce_bytes = int(nonce).to_bytes(32, "big")
    relay_hub_bytes = bytes.fromhex(RELAY_HUB.replace("0x", "").lower())
    relay_bytes = bytes.fromhex(relay_addr.replace("0x", "").lower())

    to_hash = (
        prefix
        + from_bytes
        + to_bytes
        + data
        + tx_fee_bytes
        + gas_price_bytes
        + gas_limit_bytes
        + nonce_bytes
        + relay_hub_bytes
        + relay_bytes
    )
    return keccak(to_hash)


def sign_struct_hash(struct_hash: bytes, private_key: str) -> str:
    """Sign the struct hash with eth_sign (personal_sign, adds EIP-191 prefix)."""
    msg = encode_defunct(struct_hash)
    sig = Account.sign_message(msg, private_key)
    return "0x" + sig.signature.hex()


# ─── Builder Relayer Helpers ──────────────────────────────────────────────────

def make_builder_headers(
    builder_config: BuilderConfig,
    method: str,
    path: str,
    body: dict = None,
) -> dict:
    """Generate HMAC-signed builder auth headers."""
    body_str = str(body) if body is not None else None
    payload = builder_config.generate_builder_headers(method, path, body_str)
    if payload is None:
        raise RuntimeError("Failed to generate builder auth headers")
    headers = payload.to_dict()
    headers["Content-Type"] = "application/json"
    return headers


def get_relay_payload(relayer_url: str, builder_config: BuilderConfig, eoa: str) -> dict:
    """
    GET /relay-payload?address={EOA}&type=PROXY

    Returns {"address": "0x..relay..", "nonce": "78"}
    """
    path = f"/relay-payload?address={eoa}&type=PROXY"
    headers = make_builder_headers(builder_config, "GET", path)
    resp = requests.get(f"{relayer_url}{path}", headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"relay-payload failed: {resp.status_code} {resp.text}")
    return resp.json()


def submit_proxy_transaction(
    relayer_url: str,
    builder_config: BuilderConfig,
    body: dict,
) -> dict:
    """POST /submit — submit a signed PROXY transaction to the relayer."""
    path = "/submit"
    headers = make_builder_headers(builder_config, "POST", path, body)
    resp = requests.post(
        f"{relayer_url}{path}",
        headers=headers,
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"submit failed: {resp.status_code} {resp.text}")
    return resp.json()


def poll_transaction(
    relayer_url: str,
    builder_config: BuilderConfig,
    transaction_id: str,
    target_states: set,
    fail_state: str,
    max_polls: int = 20,
    poll_interval: float = 3.0,
) -> tuple[dict | None, str | None]:
    """
    Poll /transaction?id=... until a target state or timeout.
    Returns (txn_dict, error_msg).  error_msg is set on FAILED state.
    """
    path = f"/transaction?id={transaction_id}"
    for _ in range(max_polls):
        headers = make_builder_headers(builder_config, "GET", path)
        resp = requests.get(f"{relayer_url}{path}", headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            txns = data if isinstance(data, list) else [data]
            if txns:
                txn = txns[0]
                state = txn.get("state", "")
                if state in target_states:
                    return txn, None
                if state == fail_state:
                    err = txn.get("errorMsg", "unknown error")
                    return None, err
        time.sleep(poll_interval)
    return None, "timeout"


# ─── Position Fetching ───────────────────────────────────────────────────────

def get_gamma_market(slug: str) -> dict | None:
    """
    Fetch market data from Gamma API by slug.

    Returns the first matching market dict, or None.

    IMPORTANT: The conditionId returned here is the correct on-chain CTF
    conditionId with a settled payoutDenominator.  The data-api returns a
    different (incorrect) conditionId that maps to an unsettled condition
    and causes 'payout denominator still zero' reverts.
    """
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"slug": slug},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0]
        return None
    except Exception as e:
        print(f"  WARNING: Failed to fetch Gamma market for {slug}: {e}")
        return None


def get_positions(proxy_address: str) -> list[dict]:
    """
    Fetch all open positions for a proxy wallet from data-api.

    Each position is enriched with the correct `conditionId` sourced from
    the Gamma API (the data-api conditionId is incorrect and causes reverts).
    """
    try:
        resp = requests.get(
            f"{DATA_API}/positions",
            params={"user": proxy_address, "size": 500},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data = resp.json()
        positions = data if isinstance(data, list) else []
    except Exception as e:
        print(f"  WARNING: Failed to fetch positions: {e}")
        return []

    # Enrich each position with the Gamma API conditionId
    enriched = []
    slugs_fetched: dict[str, dict | None] = {}
    for pos in positions:
        slug = pos.get("slug", "")
        if slug and slug not in slugs_fetched:
            slugs_fetched[slug] = get_gamma_market(slug)

        gamma = slugs_fetched.get(slug)
        if gamma and gamma.get("conditionId"):
            pos = dict(pos)  # copy so we don't mutate the original
            pos["conditionId"] = gamma["conditionId"]
            pos["negRisk"] = gamma.get("negRisk", pos.get("negativeRisk", False))
        enriched.append(pos)

    return enriched


# ─── Main Logic ──────────────────────────────────────────────────────────────

def main(dry_run: bool = False, redeem_all: bool = False) -> None:
    load_dotenv()

    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    funder = os.environ.get("POLYMARKET_FUNDER")
    builder_key = os.environ.get("POLYMARKET_BUILDER_KEY")
    builder_secret = os.environ.get("POLYMARKET_BUILDER_SECRET")
    builder_passphrase = os.environ.get("POLYMARKET_BUILDER_PASSPHRASE")
    relayer_url = os.environ.get("POLYMARKET_RELAYER_URL", "https://relayer-v2.polymarket.com")
    relayer_url = relayer_url.rstrip("/")

    if not private_key or not funder:
        print("ERROR: POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER must be set in .env")
        sys.exit(1)
    if not builder_key or not builder_secret or not builder_passphrase:
        print("ERROR: POLYMARKET_BUILDER_KEY/SECRET/PASSPHRASE must be set in .env")
        sys.exit(1)

    account = Account.from_key(private_key)
    eoa = account.address
    proxy = to_checksum_address(funder)

    builder_config = BuilderConfig(
        local_builder_creds=BuilderApiKeyCreds(
            key=builder_key,
            secret=builder_secret,
            passphrase=builder_passphrase,
        )
    )

    print("=" * 60)
    print("Polymarket Winnings Claim Script (Gasless via Builder Relayer)")
    print("=" * 60)
    print(f"EOA:         {eoa}")
    print(f"Proxy:       {proxy}")
    print(f"Relayer:     {relayer_url}")
    if dry_run:
        print("[DRY RUN - no transactions will be sent]")
    print()

    # ── Fetch positions ──────────────────────────────────────────────────────
    print("Fetching positions from data-api...")
    positions = get_positions(proxy)
    print(f"Found {len(positions)} open position(s)\n")

    if not positions:
        print("No open positions found. Nothing to redeem.")
        return

    # ── Categorise positions ─────────────────────────────────────────────────
    winning = []
    losing = []

    print(f"{'Outcome':6} {'Size':>10} {'Price':>7} {'Value':>8} {'Redeemable':>10}  Market")
    print("-" * 80)
    for pos in positions:
        size = float(pos.get("size", 0))
        cur_price = float(pos.get("curPrice", 0))
        redeemable = pos.get("redeemable", False)
        outcome = pos.get("outcome", "?")
        slug = pos.get("slug", "?")
        # conditionId is enriched from Gamma API in get_positions() — the
        # data-api value maps to an unsettled condition (payoutDenominator=0)
        # and causes "payout denominator still zero" reverts on-chain.
        condition_id = pos.get("conditionId", "")
        # negRisk flag comes from Gamma API (negRisk) via get_positions() enrichment
        neg_risk = pos.get("negRisk", pos.get("negativeRisk", False))
        value = size * cur_price

        print(f"{outcome:6} {size:>10.4f} {cur_price:>7.4f} ${value:>7.2f} {str(redeemable):>10}  {slug}")

        if size <= 0 or not condition_id:
            continue

        if neg_risk:
            # negRisk markets use a different adapter — skip for now
            continue

        entry = {
            "condition_id": condition_id,
            "size": size,
            "cur_price": cur_price,
            "value": value,
            "outcome": outcome,
            "slug": slug,
        }

        if cur_price >= 0.5:
            winning.append(entry)
        else:
            losing.append(entry)

    print()
    print(f"Winning positions (curPrice >= 0.5): {len(winning)}")
    print(f"Losing positions  (curPrice < 0.5):  {len(losing)}")

    to_redeem = winning + (losing if redeem_all else [])

    if not to_redeem:
        print()
        if losing and not redeem_all:
            total_losing_value = sum(p["value"] for p in losing)
            print(f"Only losing positions found (worth ${total_losing_value:.2f} total).")
            print("Use --all to redeem them anyway (burns worthless tokens).")
        else:
            print("No positions to redeem.")
            print()
            print("Possible reasons:")
            print("  - Markets haven't been resolved on-chain yet (UMA takes 1-2 hours)")
            print("  - Recent winning trades haven't appeared in data-api yet (15-30 min delay)")
            print("  - Positions were already redeemed")
        return

    # ── Execute redemptions via Builder Relayer ──────────────────────────────
    redeemed_count = 0
    total_value_redeemed = 0.0
    errors = []

    for pos in to_redeem:
        condition_id = pos["condition_id"]
        size = pos["size"]
        value = pos["value"]
        slug = pos["slug"]
        outcome = pos["outcome"]

        print()
        print(f"─── Redeeming {outcome} position on {slug} ───")
        print(f"    conditionId: {condition_id}")
        print(f"    Tokens: {size:.4f}  Value: ${value:.2f}")

        # Build calldata
        try:
            redeem_data = encode_redeem_calldata(condition_id)
            proxy_data = encode_proxy_calldata(redeem_data)
        except Exception as e:
            print(f"    ERROR encoding calldata: {e}")
            errors.append(f"{slug}: encoding error: {e}")
            continue

        if dry_run:
            print(f"    [DRY RUN] Would submit via Builder Relayer")
            print(f"    Calldata (proxy()): 0x{proxy_data.hex()[:40]}...")
            redeemed_count += 1
            total_value_redeemed += value
            continue

        # ── Get relay nonce + relay address ──────────────────────────────
        try:
            relay_payload = get_relay_payload(relayer_url, builder_config, eoa)
        except Exception as e:
            print(f"    ERROR getting relay payload: {e}")
            errors.append(f"{slug}: relay-payload error: {e}")
            continue

        nonce = relay_payload["nonce"]
        relay_addr = relay_payload["address"]
        print(f"    Relay: {relay_addr}  Nonce: {nonce}")

        # ── Sign the struct hash ──────────────────────────────────────────
        try:
            struct_hash = build_proxy_struct_hash(
                from_addr=eoa,
                proxy_factory=PROXY_FACTORY,
                data=proxy_data,
                nonce=nonce,
                relay_addr=relay_addr,
                gas_limit=DEFAULT_GAS_LIMIT,
            )
            signature = sign_struct_hash(struct_hash, private_key)
        except Exception as e:
            print(f"    ERROR signing: {e}")
            errors.append(f"{slug}: signing error: {e}")
            continue

        # ── Build and submit transaction request ──────────────────────────
        tx_body = {
            "type": "PROXY",
            "from": eoa,
            "to": PROXY_FACTORY,
            "proxyWallet": proxy,
            "data": "0x" + proxy_data.hex(),
            "nonce": str(nonce),
            "signature": signature,
            "signatureParams": {
                "gasPrice": "0",
                "gasLimit": str(DEFAULT_GAS_LIMIT),
                "relayerFee": "0",
                "relayHub": RELAY_HUB,
                "relay": relay_addr,
            },
            "metadata": "",
        }

        try:
            print(f"    Submitting to relayer...")
            resp = submit_proxy_transaction(relayer_url, builder_config, tx_body)
            transaction_id = resp.get("transactionID") or resp.get("id")
            tx_hash = resp.get("transactionHash") or resp.get("hash")
            print(f"    Relayer accepted: txID={transaction_id}  hash={tx_hash}")
        except Exception as e:
            print(f"    ERROR submitting: {e}")
            errors.append(f"{slug}: submit error: {e}")
            continue

        # ── Poll for confirmation ─────────────────────────────────────────
        if transaction_id:
            print(f"    Waiting for confirmation...")
            result, err_msg = poll_transaction(
                relayer_url, builder_config, transaction_id,
                target_states={"STATE_CONFIRMED", "STATE_MINED"},
                fail_state="STATE_FAILED",
                max_polls=20,
                poll_interval=3.0,
            )
            if result:
                confirmed_hash = result.get("transactionHash") or tx_hash
                print(f"    CONFIRMED: {confirmed_hash}")
            elif err_msg == "transaction reverted":
                print(f"    REVERTED on-chain: market not yet settled by UMA oracle.")
                print(f"    Try again in 30-60 minutes after oracle reports.")
                errors.append(f"{slug}: reverted (market not yet settled on-chain)")
                redeemed_count -= 1  # Don't count as success
                total_value_redeemed -= value
            elif err_msg == "timeout":
                print(f"    NOTE: Not yet confirmed — check relayer txID {transaction_id}")
            else:
                print(f"    FAILED: {err_msg}")
                errors.append(f"{slug}: failed — {err_msg}")
                redeemed_count -= 1
                total_value_redeemed -= value

        redeemed_count += 1
        total_value_redeemed += value

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Summary: {redeemed_count}/{len(to_redeem)} redemptions {'simulated' if dry_run else 'submitted'}")
    print(f"Total value: ${total_value_redeemed:.2f}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    print("=" * 60)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    redeem_all = "--all" in sys.argv
    main(dry_run=dry_run, redeem_all=redeem_all)
