#!/bin/bash
# Polymarket Trading Bot Daemon Startup Script
#
# This script starts the bot in the background using nohup,
# detached from the terminal session so it continues running
# even after logout.
#
# Usage:
#   ./start_bot.sh          # Start bot
#   ./start_bot.sh stop     # Stop bot
#   ./start_bot.sh status   # Check status
#   ./start_bot.sh logs     # Tail logs

cd "$(dirname "$0")"

PIDFILE="/tmp/polymarket_bot.pid"
LOGFILE="logs/bot_daemon.log"

# Create logs directory
mkdir -p logs

start_bot() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "❌ Bot is already running (PID $(cat "$PIDFILE"))"
        return 1
    fi

    echo "🚀 Starting Polymarket trading bot..."

    # Start bot with nohup, detached from terminal
    nohup python3 -u scripts/auto_trade.py >> "$LOGFILE" 2>&1 &

    # Save PID
    echo $! > "$PIDFILE"

    sleep 2

    if kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "✅ Bot started successfully (PID $(cat "$PIDFILE"))"
        echo "📋 Logs: tail -f $LOGFILE"
        echo "🛑 Stop: ./start_bot.sh stop"
    else
        echo "❌ Bot failed to start. Check logs: $LOGFILE"
        rm -f "$PIDFILE"
        return 1
    fi
}

stop_bot() {
    if [ ! -f "$PIDFILE" ]; then
        echo "⚠️  No PID file found. Bot may not be running."
        return 1
    fi

    PID=$(cat "$PIDFILE")

    if kill -0 "$PID" 2>/dev/null; then
        echo "🛑 Stopping bot (PID $PID)..."
        kill "$PID"

        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "✅ Bot stopped successfully"
                rm -f "$PIDFILE"
                return 0
            fi
            sleep 1
        done

        # Force kill if still running
        echo "⚠️  Forcing shutdown..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$PIDFILE"
        echo "✅ Bot stopped (forced)"
    else
        echo "⚠️  Bot not running (stale PID file)"
        rm -f "$PIDFILE"
    fi
}

status_bot() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "✅ Bot is running (PID $PID)"
            echo ""
            ps -p "$PID" -o pid,etime,rss,cmd
            return 0
        else
            echo "❌ Bot not running (stale PID file)"
            rm -f "$PIDFILE"
            return 1
        fi
    else
        echo "❌ Bot is not running"
        return 1
    fi
}

tail_logs() {
    if [ ! -f "$LOGFILE" ]; then
        echo "⚠️  Log file not found: $LOGFILE"
        return 1
    fi

    echo "📋 Tailing $LOGFILE (Ctrl+C to exit)"
    echo "---"
    tail -f "$LOGFILE"
}

case "${1:-start}" in
    start)
        start_bot
        ;;
    stop)
        stop_bot
        ;;
    restart)
        stop_bot
        sleep 2
        start_bot
        ;;
    status)
        status_bot
        ;;
    logs)
        tail_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
