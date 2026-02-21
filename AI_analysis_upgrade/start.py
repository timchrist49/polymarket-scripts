"""Start the AI Analysis Upgrade service with env vars from .env file."""
import os
import subprocess
import sys

# Load .env from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
env = os.environ.copy()

if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    print(f"Loaded env from {env_path}")
else:
    print(f"Warning: {env_path} not found, using system environment")

log_path = "/tmp/ai_upgrade.log"
print(f"Starting AI Analysis Upgrade service (log: {log_path})")

p = subprocess.Popen(
    [sys.executable, "-m", "AI_analysis_upgrade.service"],
    env=env,
    stdout=open(log_path, "a"),
    stderr=subprocess.STDOUT,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
print(f"Service PID: {p.pid}")
print(f"Monitor: tail -f {log_path}")
print(f"Stop: kill {p.pid}")
