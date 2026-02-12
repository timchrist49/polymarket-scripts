"""Parallel fallback fetching with racing logic."""

import asyncio
from typing import Callable, Optional, Any
import structlog

from polymarket.trading.retry_logic import fetch_with_retry

logger = structlog.get_logger()


async def fetch_with_fallbacks(
    primary_func: Callable,
    fallback_funcs: list[tuple[str, Callable]],
    validator: Optional[Callable] = None
) -> Optional[Any]:
    """
    Try primary source with retries, then race fallbacks in parallel.

    Args:
        primary_func: Primary source (already wrapped with retries)
        fallback_funcs: [(name, fetch_func), ...] for parallel execution
        validator: Optional validation function for results

    Returns:
        First successful result or None if all fail
    """

    # Step 1: Try primary (it's already wrapped with retries)
    logger.debug("Trying primary source")
    result = await primary_func()

    if result is not None:
        return result

    # Step 2: Primary failed, race fallbacks in parallel
    logger.warning("Primary source exhausted, trying fallbacks in parallel")

    tasks = [
        asyncio.create_task(fetch_with_retry(func, name))
        for name, func in fallback_funcs
    ]

    # Wait for first success or all failures
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            # Got a result! Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Validate if validator provided (for settlement)
            if validator and not validator(result):
                logger.warning("Fallback result failed validation")
                continue

            logger.info("Fallback source succeeded", source="parallel_race")
            return result

    # All sources failed
    logger.error("All sources failed (primary + fallbacks)")
    return None
