#!/usr/bin/env bash

SERVER_SCRIPT="server.py"
PID_FILE="server.pid"
LOG_FILE="server.log"
SRC_FILE="src.cpp"
SRC_BIN="./src"

echo "==============================="
echo "KRS Door Automation"
echo "==============================="

if [[ -f "$SRC_FILE" ]]; then
    echo "[INFO] Compiling $SRC_FILE..."
    g++ "$SRC_FILE" -o "$SRC_BIN" -lcurl -pthread -lgpiodcxx

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Failed to compile $SRC_FILE"
        exit 1
    fi

    chmod +x "$SRC_BIN"
    echo "[INFO] Compilation complete."
else
    echo "[WARN] $SRC_FILE not found — skipping compilation."
fi


if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "[INFO] $SERVER_SCRIPT is already running (PID $(cat $PID_FILE))."
else
    echo "[INFO] Starting $SERVER_SCRIPT..."
    nohup python3 "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"

    echo "[INFO] Server started."
    echo "[INFO] PID: $SERVER_PID"
    echo "[INFO] Logs: $LOG_FILE"
fi


if [[ -f "$SRC_BIN" ]]; then
    echo "[INFO] Running src..."
    $SRC_BIN
else
    echo "[WARN] src binary not found — skipping."
fi

echo "==============================="
echo "✔ Backend successfully launched"
echo "==============================="
