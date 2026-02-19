#!/usr/bin/env python3
"""
Claim Polymarket winnings by redeeming conditional tokens via proxy wallet.

Usage:
    python3 scripts/claim_winnings.py [--dry-run] [--all]

Options:
    --dry-run   Preview positions without sending transactions
    --all       Also redeem losing positions (worth $0, just clears them)

Requirements:
    - POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER in .env
    - EOA needs ~0.01 MATIC for gas (get it from any exchange, withdraw to Polygon)

How it works:
    1. Queries data-api.polymarket.com for open positions in proxy wallet
    2. Filters positions where curPrice > 0.5 (winning tokens worth ~$1 each)
    3. For each winning position, calls redeemPositions through the proxy wallet
       i.e.  proxy.proxy([{typeCode:1, to:CTF, value:0, data:redeemCalldata}])
    4. Signs with private key and broadcasts via Polygon RPC

Note on resolved status:
    Markets must be fully resolved on-chain before tokens can be redeemed.
    Polymarket's settler (UMA) typically reports payouts within 1-2 hours of market close.
    If curPrice shows 1.0 but redeem fails, the market may not be fully settled yet.
"""

import os
import sys
import json
import time

import requests
from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address
from eth_account import Account
from dotenv import load_dotenv

# ─── Constants ───────────────────────────────────────────────────────────────

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CHAIN_ID = 137  # Polygon Mainnet
GAS_PRICE_GWEI = 35  # 35 gwei, sufficient for Polygon
MIN_MATIC_REQUIRED = 0.005  # ~$0.005 at current prices, enough for 3+ redemptions

DATA_API = "https://data-api.polymarket.com"
POLYGON_RPCS = [
    "https://1rpc.io/matic",
    "https://rpc-mainnet.maticvigil.com",
    "https://polygon.llamarpc.com",
]

# ─── ABI Encoding ────────────────────────────────────────────────────────────

# Function selectors
REDEEM_SELECTOR = keccak(b"redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
PROXY_SELECTOR = keccak(b"proxy((uint8,address,uint256,bytes)[])")[:4]


def encode_redeem_calldata(condition_id_hex: str) -> bytes:
    """
    Encode redeemPositions(USDC, bytes32(0), conditionId, [1,2]) calldata.

    Args:
        condition_id_hex: conditionId as hex string (with or without 0x prefix)

    Returns:
        ABI-encoded calldata bytes
    """
    condition_id = bytes.fromhex(condition_id_hex.replace("0x", ""))
    args = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [USDC_ADDRESS, b"\x00" * 32, condition_id, [1, 2]],
    )
    return REDEEM_SELECTOR + args


def encode_proxy_calldata(inner_calldata: bytes) -> bytes:
    """
    Encode proxy([(typeCode=1, to=CTF, value=0, data=calldata)]) calldata.

    The proxy wallet's proxy() function executes calls on behalf of the owner.
    typeCode=1 means standard call (not delegatecall).

    Args:
        inner_calldata: The encoded calldata to execute inside the proxy call

    Returns:
        ABI-encoded calldata bytes for the proxy() call
    """
    args = abi_encode(
        ["(uint8,address,uint256,bytes)[]"],
        [[(1, CTF_ADDRESS, 0, inner_calldata)]],
    )
    return PROXY_SELECTOR + args


# ─── RPC Helpers ─────────────────────────────────────────────────────────────

_working_rpc = None


def rpc_call(method: str, params: list) -> dict:
    """Make a JSON-RPC call, trying multiple Polygon RPC endpoints."""
    global _working_rpc

    rpcs = [_working_rpc] + POLYGON_RPCS if _working_rpc else POLYGON_RPCS
    rpcs = [r for r in rpcs if r]  # remove None

    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}

    for rpc in rpcs:
        try:
            resp = requests.post(rpc, json=payload, timeout=10)
            data = resp.json()
            if "result" in data:
                _working_rpc = rpc
                return data
            if "error" in data and "Unauthorized" not in str(data["error"]):
                return data  # Return real errors (not auth failures)
        except Exception:
            continue

    return {"error": f"All RPCs failed for {method}"}


def get_matic_balance(address: str) -> float:
    """Get MATIC balance of address in MATIC units."""
    resp = rpc_call("eth_getBalance", [address, "latest"])
    if "result" not in resp:
        return 0.0
    return int(resp["result"], 16) / 1e18


def get_nonce(address: str) -> int:
    """Get current transaction nonce for address."""
    resp = rpc_call("eth_getTransactionCount", [address, "latest"])
    if "result" not in resp:
        raise RuntimeError(f"Failed to get nonce: {resp}")
    return int(resp["result"], 16)


def estimate_gas(from_addr: str, to_addr: str, data: bytes) -> int:
    """Estimate gas for a transaction."""
    resp = rpc_call(
        "eth_estimateGas",
        [{"from": from_addr, "to": to_addr, "data": "0x" + data.hex()}],
    )
    if "result" not in resp:
        # Return a safe default if estimation fails
        return 200_000
    return int(int(resp["result"], 16) * 1.2)  # 20% buffer


def send_raw_transaction(raw_tx_hex: str) -> str:
    """Broadcast a signed raw transaction. Returns tx hash."""
    resp = rpc_call("eth_sendRawTransaction", [raw_tx_hex])
    if "result" in resp:
        return resp["result"]
    raise RuntimeError(f"Transaction rejected: {resp.get('error', resp)}")


def wait_for_receipt(tx_hash: str, timeout: int = 60) -> dict:
    """Poll for transaction receipt until confirmed or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = rpc_call("eth_getTransactionReceipt", [tx_hash])
        receipt = resp.get("result")
        if receipt:
            return receipt
        time.sleep(3)
    return {}


# ─── Position Fetching ───────────────────────────────────────────────────────

def get_positions(proxy_address: str) -> list[dict]:
    """
    Fetch all open positions for a proxy wallet from data-api.

    Returns list of position dicts with keys:
        conditionId, size, curPrice, redeemable, outcome, slug, negativeRisk, etc.
    """
    try:
        resp = requests.get(
            f"{DATA_API}/positions",
            params={"user": proxy_address, "size": 500},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  WARNING: Failed to fetch positions: {e}")
        return []


# ─── Main Logic ──────────────────────────────────────────────────────────────

def main(dry_run: bool = False, redeem_all: bool = False) -> None:
    load_dotenv()

    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    funder = os.environ.get("POLYMARKET_FUNDER")

    if not private_key or not funder:
        print("ERROR: POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER must be set in .env")
        sys.exit(1)

    account = Account.from_key(private_key)
    eoa = account.address
    proxy = to_checksum_address(funder)

    print("=" * 60)
    print("Polymarket Winnings Claim Script")
    print("=" * 60)
    print(f"EOA:   {eoa}")
    print(f"Proxy: {proxy}")
    if dry_run:
        print("[DRY RUN - no transactions will be sent]")
    print()

    # ── Check MATIC balance ──────────────────────────────────────────────────
    matic = get_matic_balance(eoa)
    print(f"MATIC balance: {matic:.6f} MATIC")

    if matic < MIN_MATIC_REQUIRED and not dry_run:
        print(f"\nERROR: Insufficient MATIC for gas.")
        print(f"  Need:    {MIN_MATIC_REQUIRED:.4f} MATIC (~${MIN_MATIC_REQUIRED * 0.8:.4f} USD)")
        print(f"  Current: {matic:.6f} MATIC")
        print()
        print("How to get MATIC (very cheap, ~$0.01 is enough):")
        print("  1. Buy on Binance, Coinbase, etc. and withdraw to Polygon network")
        print("  2. Bridge from Ethereum: https://wallet.polygon.technology")
        print("  3. Use a faucet: https://faucet.polygon.technology")
        print()
        print(f"  Send to: {eoa}")
        print()
        print("Run with --dry-run to preview positions without sending transactions.")
        sys.exit(1)

    # ── Fetch positions ──────────────────────────────────────────────────────
    print("Fetching positions from data-api...")
    positions = get_positions(proxy)
    print(f"Found {len(positions)} open position(s)\n")

    if not positions:
        print("No open positions found. Nothing to redeem.")
        return

    # ── Categorize positions ─────────────────────────────────────────────────
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
        condition_id = pos.get("conditionId", "")
        neg_risk = pos.get("negativeRisk", False)
        value = size * cur_price

        print(f"{outcome:6} {size:>10.4f} {cur_price:>7.4f} ${value:>7.2f} {str(redeemable):>10}  {slug}")

        if size <= 0 or not condition_id:
            continue

        if neg_risk:
            # Negative risk markets use a different redemption flow (negRiskAdapter)
            # Skip for now — rare for BTC 15-min markets
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
            print("Use --all to redeem them anyway (burns worthless tokens, costs gas).")
        else:
            print("No positions to redeem.")
            print()
            print("Possible reasons:")
            print("  - Markets haven't been resolved on-chain yet (UMA takes 1-2 hours)")
            print("  - Recent winning trades haven't appeared in data-api yet (15-30 min delay)")
            print("  - Positions were already redeemed")
        return

    # ── Execute redemptions ──────────────────────────────────────────────────
    if not dry_run:
        nonce = get_nonce(eoa)
        gas_price = GAS_PRICE_GWEI * 10**9
    else:
        nonce = 0
        gas_price = GAS_PRICE_GWEI * 10**9

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

        # Estimate gas
        gas_limit = estimate_gas(eoa, proxy, proxy_data)
        gas_cost_matic = (gas_limit * gas_price) / 1e18
        print(f"    Gas: {gas_limit:,} units @ {GAS_PRICE_GWEI} gwei = {gas_cost_matic:.6f} MATIC")

        if dry_run:
            print(f"    [DRY RUN] Would send transaction")
            print(f"    Calldata: 0x{proxy_data.hex()[:40]}...")
            redeemed_count += 1
            total_value_redeemed += value
            continue

        # Build and sign transaction
        tx = {
            "nonce": nonce,
            "gasPrice": gas_price,
            "gas": gas_limit,
            "to": proxy,
            "value": 0,
            "data": proxy_data,
            "chainId": CHAIN_ID,
        }

        try:
            signed = account.sign_transaction(tx)
            raw_hex = "0x" + signed.raw_transaction.hex()

            print(f"    Sending transaction (nonce={nonce})...")
            tx_hash = send_raw_transaction(raw_hex)
            print(f"    TX hash: {tx_hash}")

            # Wait for confirmation
            print(f"    Waiting for confirmation...")
            receipt = wait_for_receipt(tx_hash, timeout=90)

            if receipt:
                status = int(receipt.get("status", "0x0"), 16)
                gas_used = int(receipt.get("gasUsed", "0x0"), 16)
                if status == 1:
                    print(f"    SUCCESS! Gas used: {gas_used:,}")
                    redeemed_count += 1
                    total_value_redeemed += value
                else:
                    print(f"    FAILED (tx reverted). Market may not be fully settled yet.")
                    errors.append(f"{slug}: tx reverted (market not settled?)")
            else:
                print(f"    TIMEOUT waiting for receipt. Check: https://polygonscan.com/tx/{tx_hash}")
                redeemed_count += 1
                total_value_redeemed += value

            nonce += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append(f"{slug}: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Summary: {redeemed_count}/{len(to_redeem)} redemptions {'simulated' if dry_run else 'completed'}")
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
