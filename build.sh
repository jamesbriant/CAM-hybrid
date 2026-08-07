#!/bin/bash
#SBATCH --job-name=cesm_build
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

USERNAME="jamesbriant" # Your docker hub username
TAG="latest"

# Path to your "key"
AUTH="--authfile $HOME/my_docker_auth.json"

# --- 1. BUILD BASE ---
podman-hpc build -t $USERNAME/cesm_base:$TAG -f cesm_base/Dockerfile .
podman-hpc push $AUTH $USERNAME/cesm_base:$TAG

# --- 2. BUILD CESM ---
# We run this from the root so "COPY cesm/crt/..." in the Dockerfile would work
# OR if your Dockerfile says "COPY crt/...", we run it like this:
podman-hpc build -t $USERNAME/cesm:$TAG -f cesm/Dockerfile cesm/
podman-hpc push $AUTH $USERNAME/cesm:$TAG

# --- 3. BUILD FTORCH ---
podman-hpc build -t $USERNAME/cesm_ftorch:$TAG -f cesm_ftorch/Dockerfile cesm_ftorch/
podman-hpc push $AUTH $USERNAME/cesm_ftorch:$TAG

echo "Build and Push Complete!"
