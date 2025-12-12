#!/usr/bin/env bash

echo "=============================================="
echo " DOCKER DEPLOY "
echo "=============================================="

# normalize input argument (cpu/gpu/jetson/pi)
INPUT=${1,,}  
case $INPUT in
  cpu)    TARGET="cpu-x86";   PROFILE="cpu" ;;
  gpu)    TARGET="cuda-x86";  PROFILE="gpu" ;;
  pi)     TARGET="pi";        PROFILE="pi" ;;
  *)
    echo "Invalid argument. Usage: ./deploy.sh {cpu|gpu|pi}"
    exit 1
    ;;
esac

echo "[INFO] Selected platform: $INPUT"
echo "[INFO] Docker TARGET     : $TARGET"
echo "[INFO] Compose PROFILE   : $PROFILE"

# detect host OS (Raspberry Pi requires different Docker install)
if grep -qi "raspbian" /etc/os-release || grep -qi "raspberry" /etc/os-release; then
    OS="raspberry"
else
    OS="ubuntu"
fi
echo "[INFO] Detected OS: $OS"

# install Docker if missing
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
    echo "[INFO] Docker already installed"
fi

# ensure Docker Compose plugin exists
if ! docker compose version &>/dev/null; then
    echo "[INFO] Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
else
    echo "[INFO] Docker Compose plugin already installed"
fi

# ensure user has permission to use Docker without sudo
if ! groups $USER | grep -q "\bdocker\b"; then
    echo "[INFO] Adding user to docker and video groups..."
    sudo usermod -aG docker $USER
    sudo usermod -aG video $USER
    sudo usermod -a -G gpio $USER
    echo "Please log out and log back in to apply permissions."
fi

echo "----------------------------------------------"
echo "                    🐋                        "
echo "----------------------------------------------"

docker compose down --remove-orphans 2>/dev/null

# pass TARGET to Docker build
TARGET=$TARGET docker compose --profile $PROFILE up -d --build
if [ $? -ne 0 ]; then
    echo "[ERROR] Build/run failed"
    exit 1
fi

echo "[INFO] Container built and started successfully!"

# auto-open shell inside the container
CONTAINER_NAME="door-automation"
echo "[INFO] Waiting for container to initialize..."
sleep 2

echo "[INFO] Opening interactive shell..."
xhost +local:docker
docker exec -it $CONTAINER_NAME bash
