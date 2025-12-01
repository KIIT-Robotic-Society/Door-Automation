#!/usr/bin/env bash

echo "=============================================="
echo " DOCKER DEPLOY "
echo "=============================================="

INPUT=${1,,}  
case $INPUT in
  cpu)    TARGET="cpu-x86";   PROFILE="cpu" ;;
  gpu)    TARGET="cuda-x86";  PROFILE="gpu" ;;
  jetson) TARGET="jetson";    PROFILE="jetson" ;;
  pi)     TARGET="pi";        PROFILE="pi" ;;
  *)
    echo "Invalid argument."
    echo "Usage:"
    echo " ./deploy.sh cpu"
    echo " ./deploy.sh gpu"
    echo " ./deploy.sh jetson"
    echo " ./deploy.sh pi"
    exit 1
    ;;
esac

echo "[INFO] Selected platform: $INPUT"
echo "[INFO] Docker TARGET     : $TARGET"
echo "[INFO] Compose PROFILE   : $PROFILE"

if grep -qi "raspbian" /etc/os-release || grep -qi "raspberry" /etc/os-release; then
    OS="raspberry"
else
    OS="ubuntu"
fi

echo "[INFO] Detected OS: $OS"

if ! command -v docker &>/dev/null; then
    echo "[INFO] Docker not installed → installing..."
    if [ "$OS" = "raspberry" ]; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
    else
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    fi
else
    echo "Docker already installed"
fi


if ! docker compose version &>/dev/null; then
    echo "[INFO] Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
else
    echo "Docker Compose plugin already installed"
fi


if ! groups $USER | grep -q "\bdocker\b"; then
    echo "[INFO] Adding user to docker group..."
    sudo usermod -aG docker $USER
    sudo usermod -aG video $USER
    echo "Logout & login again after this script (permission will apply)."
fi


echo "----------------------------------------------"
echo "🐋 Building and running container..."
echo "----------------------------------------------"
docker compose down --remove-orphans 2>/dev/null

TARGET=$TARGET docker compose --profile $PROFILE up -d --build
if [ $? -ne 0 ]; then
    echo "Build/run failed"
    exit 1
fi

echo "Container built and started!"


CONTAINER_NAME="door-automation"
echo "⌛ Waiting for container to initialize..."
sleep 2

echo "➡ Entering container..."
docker exec -it $CONTAINER_NAME bash
