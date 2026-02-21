# AI Analysis Upgrade — Shadow Service

Runs enhanced multi-timeframe AI analysis in parallel with the production bot.
Compares v1 (current) vs v2 (new) predictions after each market settles and sends Telegram alerts.

## Architecture

- **Read-only against production** — reads `ai_analysis_log`, never modifies production tables
- **Writes to** `ai_analysis_v2` table in the same `data/performance.db`
- **Two loops** running every 30s:
  1. Analysis loop: find new v1 rows → run v2 analysis → save
  2. Alert loop: find settled markets → send Telegram comparison

## Start

```bash
cd /root/polymarket-scripts
python3 AI_analysis_upgrade/start.py
```

Or manually:
```bash
cd /root/polymarket-scripts
python3 -c "
import subprocess, os
env = os.environ.copy()
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
p = subprocess.Popen(['python3', '-m', 'AI_analysis_upgrade.service'], env=env,
    stdout=open('/tmp/ai_upgrade.log', 'a'), stderr=subprocess.STDOUT)
print('PID:', p.pid)
"
```

## Stop

```bash
pkill -f "AI_analysis_upgrade.service"
```

## Monitor

```bash
tail -f /tmp/ai_upgrade.log
```

## What it reads

- `data/performance.db` → `ai_analysis_log` (v1 predictions from production bot)

## What it writes

- `data/performance.db` → `ai_analysis_v2` (v2 predictions + comparison results)

## Query results after 50+ markets

```python
import sqlite3
conn = sqlite3.connect("/root/polymarket-scripts/data/performance.db")
for row in conn.execute("""
    SELECT COUNT(*) total,
           SUM(v1_correct) v1_wins,
           SUM(v2_correct) v2_wins,
           ROUND(100.0*SUM(v2_correct)/COUNT(*),1) v2_pct
    FROM ai_analysis_v2
    WHERE actual_outcome IS NOT NULL
"""):
    print(f"Total: {row[0]} | V1: {row[1]}/{row[0]} | V2: {row[2]}/{row[0]} ({row[3]}%)")
```

## Known limitations

- Kraken OHLCV may fail on this server (DNS restriction) — service falls back to neutral trend score
- CLOB REST may fail for expired markets — service falls back to 0.50 odds
- CoinGecko PRO signals require a valid API key in `.env`
