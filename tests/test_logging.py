# tests/test_logging.py
import logging
import io
import sys
from polymarket.utils.logging import setup_logging, get_logger

def test_setup_logging_creates_logger():
    """Test setup_logging creates a logger."""
    logger = setup_logging("INFO", json=False)
    assert logger is not None
    assert logger.level == logging.INFO

def test_get_logger_returns_same_instance():
    """Test get_logger returns same logger for same name."""
    logger1 = get_logger("test")
    logger2 = get_logger("test")
    assert logger1 is logger2

def test_logger_output_format(capsys):
    """Test logger output format."""
    setup_logging("INFO", json=False)
    logger = get_logger("test")
    logger.info("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out
