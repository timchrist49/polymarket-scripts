"""Retry logic with exponential backoff for API calls."""

import asyncio
from typing import Callable, Optional, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3          # Try 3 times total (1 initial + 2 retries)
    initial_delay: float = 2.0     # Start with 2 second delay
    backoff_factor: float = 2.0    # Double delay each retry
    timeout: int = 30              # 30 seconds per attempt


async def fetch_with_retry(
    fetch_func: Callable,
    source_name: str,
    config: RetryConfig = None
) -> Optional[Any]:
    """
    Generic retry wrapper for any fetch function.

    Args:
        fetch_func: Async function to call
        source_name: Name for logging
        config: Retry configuration

    Returns:
        Result from fetch_func or None if all attempts fail
    """
    if config is None:
        config = RetryConfig()

    last_error = None

    for attempt in range(config.max_attempts):
        try:
            # Attempt fetch with timeout
            result = await asyncio.wait_for(
                fetch_func(),
                timeout=config.timeout
            )

            # Log success if this was a retry
            if attempt > 0:
                logger.info(
                    f"{source_name} succeeded on retry",
                    attempt=attempt + 1
                )

            return result

        except asyncio.TimeoutError:
            last_error = f"{source_name} timeout after {config.timeout}s"

        except Exception as e:
            last_error = f"{source_name} error: {str(e)}"

        # Don't delay after last attempt
        if attempt < config.max_attempts - 1:
            delay = config.initial_delay * (config.backoff_factor ** attempt)
            logger.warning(
                f"{source_name} failed, retrying",
                attempt=attempt + 1,
                delay_seconds=delay,
                error=last_error
            )
            await asyncio.sleep(delay)

    # All attempts failed
    logger.error(f"{source_name} failed all retries", error=last_error)
    return None
