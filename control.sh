#!/usr/bin/env bash

SERVER_SCRIPT="server.py"
PID_FILE="server.pid"
LOG_FILE="server.log"

# Check if PID file exists 
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "[INFO] $SERVER_SCRIPT is already running (PID $(cat $PID_FILE))."
else
    echo "[INFO] Starting $SERVER_SCRIPT..."

    nohup python3 "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    echo $SERVER_PID > "$PID_FILE"

    echo "[INFO] Server started successfully."
    echo "[INFO] PID: $SERVER_PID"
    echo "[INFO] Logs: $LOG_FILE"
fi

echo "[INFO] Running Src"
./src

echo
echo "[INFO] ./src finished execution."
echo "--------------------------------------"
