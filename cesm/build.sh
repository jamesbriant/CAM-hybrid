#!/bin/bash
#SBATCH --job-name=podman_build
#SBATCH --partition=gh  # Use the Grace-Hopper partition for AArch64 builds
#SBATCH --nodes=1
#SBATCH --time=02:00:00

# 1. Build the image on the compute node
podman-hpc build --ulimit nofile=3950:4000 -t cesm .

# 2. Migrate the image to persistent shared scratch
# This is CRITICAL. Without this, the image disappears when the job ends.
podman-hpc migrate cesm

# 3. Verify it is now in the shared (R/O=true) storage
podman-hpc images
