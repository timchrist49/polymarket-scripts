# Polymarket OpenClaw Integration Design

**Date:** 2026-02-09
**Author:** Claude Code
**Status:** Approved

## Goal

Create comprehensive documentation and OpenClaw agent integration for the Polymarket BTC 15-minute trading scripts, then push to GitHub.

## Requirements Summary

- **Documentation Format:** Both - comprehensive approach (inline docstrings + separate markdown docs)
- **Detail Level:** Layered (usage → API → internals)
- **Repo Structure:** Agent-optimized with `openclaw/` folder
- **Security:** Basic (gitignore + manual review)

## Repository Structure

```
polymarket-scripts/
├── README.md                          # User guide (overview, quick start)
├── .env.example                       # Credential template
├── .gitignore                         # Exclude .env, __pycache__, etc.
├── requirements.txt                   # Python dependencies
│
├── scripts/                           # Executable scripts (agent-facing)
│   ├── fetch_markets.py              # Full docstrings
│   ├── place_order.py                # Full docstrings
│   └── portfolio_status.py           # Full docstrings
│
├── polymarket/                        # Core library (internal API)
│   ├── __init__.py
│   ├── client.py                     # Main API client
│   ├── auth.py                       # Authentication
│   ├── config.py                     # Configuration
│   ├── models.py                     # Data models
│   ├── exceptions.py                 # Custom exceptions
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── retry.py
│
├── tests/                            # Test suite
│
└── openclaw/                         # AGENT-OPTIMIZED (new)
    ├── README.md                     # Agent quick-start guide
    ├── prompts/                      # Reusable prompt templates
    │   ├── fetch_markets.txt
    │   ├── place_order.txt
    │   └── portfolio_check.txt
    ├── skills/                       # Skill definitions for agents
    │   ├── polymarket_fetch.yaml
    │   ├── polymarket_trade.yaml
    │   └── polymarket_status.yaml
    └── examples/                     # Sample agent workflows
        ├── daily_trading_flow.md
        └── monitoring_workflow.md
```

## Documentation Standards

### Layer 1 - User Level (README + openclaw/README.md)
- Purpose: What does this do?
- Prerequisites: What's needed to run it?
- Quick start: Copy-paste commands
- Common use cases: Real-world examples
- Troubleshooting: Common errors and fixes

### Layer 2 - API Level (Script docstrings)
```python
"""
Script purpose and description.

Usage:
    python scripts/script.py --arg value

Arguments:
    --arg: Description

Returns:
    Description of return values

Examples:
    Command examples

Exit codes:
    0: Success
    1: Error
"""
```

### Layer 3 - Implementation Level (docs/architecture.md)
- Architecture diagrams (data flow, auth flow)
- Algorithm explanations (market discovery, retry logic)
- Design decisions (why FOK orders, why signature_type=1)

## Security Checklist

Before pushing to GitHub:
1. Verify `.gitignore` excludes `.env`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.claude/`, `*.log`
2. Manual code review for hardcoded credentials
3. Verify `.env.example` has placeholders only

## Git Setup

```bash
git config user.name "timchrist49"
git config user.email "timothy.christ49@gmail.com"
git remote add origin https://github.com/timchrist49/polymarket-scripts.git
git push -u origin master
```

## New Files to Create

1. `openclaw/README.md` - Agent quick start guide
2. `openclaw/prompts/fetch_markets.txt` - Fetch market prompt template
3. `openclaw/prompts/place_order.txt` - Place order prompt template
4. `openclaw/prompts/portfolio_check.txt` - Portfolio status prompt template
5. `openclaw/skills/polymarket_fetch.yaml` - Fetch skill definition
6. `openclaw/skills/polymarket_trade.yaml` - Trade skill definition
7. `openclaw/skills/polymarket_status.yaml` - Status skill definition
8. `openclaw/examples/daily_trading_flow.md` - Trading workflow example
9. `openclaw/examples/monitoring_workflow.md` - Monitoring workflow example
10. `docs/architecture.md` - Architecture documentation

## Files to Update

Add comprehensive docstrings to:
1. `scripts/fetch_markets.py`
2. `scripts/place_order.py`
3. `scripts/portfolio_status.py`
4. `polymarket/client.py`
5. `polymarket/auth.py`
6. `polymarket/config.py`
7. `polymarket/models.py`
8. `polymarket/utils/logging.py`
9. `polymarket/utils/retry.py`
10. `README.md` - Update with openclaw section
