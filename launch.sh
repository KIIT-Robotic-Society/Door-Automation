#!/usr/bin/env bash


SERVER_SCRIPT="server.py"     # python API / backend script
PID_FILE="server.pid"         # stores running server PID
LOG_FILE="server.log"         # log output for server

BUILD_DIR="build"             # Cmake build directory
BIN_NAME="Door-Automation"    # output binary name
BIN_PATH="./$BUILD_DIR/$BIN_NAME"

echo "==============================="
echo "KRS Door Automation"
echo "==============================="

# build C++ Project using CMake
echo "[INFO] Rebuilding project using CMake..."

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..
make -j$(nproc)    # parallel build using all CPU cores

# check if build succeeded
if [[ $? -ne 0 ]]; then
    echo "[ERROR] C++ build failed!"
    exit 1
fi

cd ..
chmod +x "$BIN_PATH"

echo "[INFO] Build complete: $BIN_PATH"


# start Python server (FastAPI / backend)
# only start if not already running
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    # server already active
    echo "[INFO] $SERVER_SCRIPT already running (PID $(cat "$PID_FILE"))."
else
    # launch server in background
    echo "[INFO] Starting $SERVER_SCRIPT..."
    nohup python3 "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    echo "[INFO] Server started with PID $SERVER_PID"
    echo "[INFO] Logs: $LOG_FILE"
fi


# run the C++ Door Automation System
echo "[INFO] Running Door Automation System..."
$BIN_PATH
EXIT_CODE=$?

echo "[INFO] Program exited with status $EXIT_CODE"

echo "[INFO] Program exited with status $EXIT_CODE"
