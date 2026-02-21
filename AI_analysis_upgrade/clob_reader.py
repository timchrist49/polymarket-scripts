"""Fetch real-time YES/NO odds from Polymarket CLOB REST API.

This uses the same public endpoints as the production bot.
"""
import json
import aiohttp
import structlog
from typing import Optional

from AI_analysis_upgrade import config

logger = structlog.get_logger()


async def fetch_clob_odds(market_slug: str, session: aiohttp.ClientSession) -> dict:
    """Fetch current YES/NO odds for a market.

    Returns:
        {"yes": float, "no": float} or {"yes": 0.5, "no": 0.5} on failure.
    """
    # Step 1: Get token IDs from Gamma API
    try:
        gamma_url = f"{config.GAMMA_API_URL}/markets"
        async with session.get(
            gamma_url,
            params={"slug": market_slug},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            markets = await resp.json()
            if not markets:
                return {"yes": 0.5, "no": 0.5}
            market = markets[0]
            # Gamma returns clobTokenIds as either a list or a JSON-encoded string
            token_ids = market.get("clobTokenIds", [])
            if isinstance(token_ids, str):
                try:
                    token_ids = json.loads(token_ids)
                except (json.JSONDecodeError, ValueError):
                    token_ids = []
            if len(token_ids) < 2:
                return {"yes": 0.5, "no": 0.5}
            yes_token_id = token_ids[0]
            no_token_id = token_ids[1]
    except Exception as e:
        logger.warning("fetch_clob_odds: Gamma lookup failed", slug=market_slug, error=str(e))
        return {"yes": 0.5, "no": 0.5}

    # Step 2: Get best bid/ask from CLOB order book
    try:
        clob_url = f"{config.CLOB_REST_URL}/book"
        yes_price, no_price = 0.5, 0.5

        for token_id, key in ((yes_token_id, "yes"), (no_token_id, "no")):
            async with session.get(
                clob_url,
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                book = await resp.json()
                bids = book.get("bids", [])
                if bids:
                    best_bid = float(bids[0]["price"])
                    if key == "yes":
                        yes_price = best_bid
                    else:
                        no_price = best_bid

        return {"yes": yes_price, "no": no_price}
    except Exception as e:
        logger.warning("fetch_clob_odds: CLOB lookup failed", slug=market_slug, error=str(e))
        return {"yes": 0.5, "no": 0.5}
