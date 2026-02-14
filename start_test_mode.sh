#!/bin/bash
export TEST_MODE=true
cd /root/polymarket-scripts
exec python3 scripts/auto_trade.py
