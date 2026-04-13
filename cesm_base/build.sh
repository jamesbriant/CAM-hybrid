#!/bin/bash
#SBATCH --job-name=podman_build
#SBATCH --partition=gh  # Use the Grace-Hopper partition for AArch64 builds
#SBATCH --nodes=1
#SBATCH --time=02:00:00

# 1. Build the image on the compute node
# You can now safely use a higher N_PROC
podman-hpc build --build-arg N_PROC=$(nproc) -t cesm_base .

# 2. Migrate the image to persistent shared scratch
# This is CRITICAL. Without this, the image disappears when the job ends.
podman-hpc migrate cesm_base

# 3. Verify it is now in the shared (R/O=true) storage
podman-hpc images
