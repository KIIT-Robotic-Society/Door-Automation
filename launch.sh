#!/usr/bin/env bash

set -e

SERVER_SCRIPT="server.py"
PID_FILE="server.pid"
LOG_FILE="server.log"

BUILD_DIR="build"
BIN_NAME="Door-Automation"
BIN_PATH="./$BUILD_DIR/$BIN_NAME"

API_URL="http://127.0.0.1:8000/heartbeat"
API_KEY="uBJjvkPOIFJguPO"

echo "===================================="
echo "        KRS Door Automation         "
echo "===================================="


# 1) START PYTHON FASTAPI SERVER
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[INFO] Server already running (PID $(cat "$PID_FILE"))"
else
    echo "[INFO] Starting Python FastAPI server..."
    nohup python3 "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    echo "[INFO] Server started with PID $SERVER_PID"
fi

#TIMER FOR API TO BE READY
echo -n "[INFO] Waiting for API to become ready"

API_READY=0
MAX_RETRIES=40   # 40 seconds max wait

for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s -H "x-api-key: $API_KEY" "$API_URL" >/dev/null; then
        echo -e "\n[INFO] API is live and responding!"
        API_READY=1
        break
    fi
    echo -n "."
    sleep 1
done

if [[ $API_READY -ne 1 ]]; then
    echo -e "\n[ERROR] API did NOT start after $MAX_RETRIES seconds."
    echo "[ERROR] Check server.log for Python errors."
    exit 1
fi

# 3) CMAKE BUILD (ONLY AFTER API IS ONLINE)
echo "[INFO] Building C++ project with CMake..."

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. >/dev/null
make -j$(nproc)

cd ..
chmod +x "$BIN_PATH"

echo "[INFO] Build complete: $BIN_PATH"

#RUN THE C++ DOOR AUTOMATION BINARY
echo "[INFO] Running Door Automation System..."
$BIN_PATH
EXIT_CODE=$?

echo "[INFO] Program exited with status $EXIT_CODE"
exit $EXIT_CODE
