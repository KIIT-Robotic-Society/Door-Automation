#!/usr/bin/env bash

SERVER_SCRIPT="server.py"
PID_FILE="server.pid"
LOG_FILE="server.log"

BUILD_DIR="build"
BIN_NAME="Door-Automation"
BIN_PATH="./$BUILD_DIR/$BIN_NAME"

echo "==============================="
echo "KRS Door Automation"
echo "==============================="

echo "[INFO] Rebuilding project using CMake..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"


cmake ..

make -j$(nproc)
if [[ $? -ne 0 ]]; then
    echo "[ERROR] C++ build failed!"
    exit 1
fi

cd ..
chmod +x "$BIN_PATH"
echo "[INFO] Build complete: $BIN_PATH"


if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[INFO] $SERVER_SCRIPT already running (PID $(cat "$PID_FILE"))."
else
    echo "[INFO] Starting $SERVER_SCRIPT..."
    nohup python3 "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    echo "[INFO] Server started with PID $SERVER_PID"
    echo "[INFO] Logs: $LOG_FILE"
fi

echo "[INFO] Running Door Automation System..."
$BIN_PATH
EXIT_CODE=$?

echo "[INFO] Program exited with status $EXIT_CODE"
