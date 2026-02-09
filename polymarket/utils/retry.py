# polymarket/utils/retry.py
"""Exponential backoff retry decorator."""

import time
import random
import functools
from typing import Callable, Type
from polymarket.exceptions import PolymarketError


def retry(
    exceptions: type[PolymarketError] | tuple[type[PolymarketError], ...] = PolymarketError,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        exceptions: Exception type(s) to catch and retry on
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        jitter: Add random jitter to delay to prevent thundering herd

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed, raise the exception
                        break

                    # Calculate delay with optional jitter
                    actual_delay = delay
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())

                    # Simple logger (avoid circular import)
                    import sys
                    print(
                        f"Retry {attempt + 1}/{max_attempts} after {actual_delay:.2f}s: {e}",
                        file=sys.stderr,
                    )

                    time.sleep(actual_delay)
                    delay *= backoff_factor

            # All attempts exhausted
            raise last_exception

        return wrapper

    return decorator
