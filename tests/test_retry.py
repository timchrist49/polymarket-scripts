# tests/test_retry.py
import pytest
from polymarket.utils.retry import retry
from polymarket.exceptions import RateLimitError, UpstreamAPIError

def test_retry_success_on_first_try():
    """Test function that succeeds immediately."""
    @retry(max_attempts=3, initial_delay=0.01)
    def succeeds():
        return "success"

    assert succeeds() == "success"

def test_retry_success_after_failure():
    """Test function that succeeds after retries."""
    attempts = [0]

    @retry(max_attempts=3, initial_delay=0.01)
    def fails_then_succeeds():
        attempts[0] += 1
        if attempts[0] < 2:
            raise RateLimitError("try again")
        return "success"

    assert fails_then_succeeds() == "success"
    assert attempts[0] == 2

def test_retry_exhausted():
    """Test function that never succeeds."""
    @retry(max_attempts=3, initial_delay=0.01)
    def always_fails():
        raise UpstreamAPIError("server error")

    with pytest.raises(UpstreamAPIError, match="server error"):
        always_fails()
