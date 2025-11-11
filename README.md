# CAM-hybrid

See the Medium article by Scotty for an outline of this project's workflow.
https://medium.com/@twins.corgi.0a/hybrid-ai-hpc-workflows-integrating-pytorch-with-cesm-on-nvidia-gh200-94a0bc34ffd9

## Installation

```bash
git clone ...
```

## GP Training

Train the Gaussian process model before proceeding to running the hybrid model.

See [ENTER REPO URL HERE]

Create a new directory and move the trained models to `models/your_trained_model.pt`.

## Running the Model

### Step 1: Build Containers

There are 3 containers to build sequentially. 

1. `cesm_base`
2. `cesm`
3. `cesm_ftorch`

I used podman for building on HPC but docker and singularity are alternatives.

### Step 2: Create cases

First update `HOST_BASE_DIR` variable near the top of `manage_cesm.sh`. This script is the hub from which cases are created and simulations are launched.

```bash

```

### Step 3: Launch

## CAM Changes

Only two file changes are made to the CAM codebase.

`cam/src/physics/cam/cam_gp.F90` is a new file containing the FTorch forward pass (GP prediction) implementation.

`cam/src/physics/cam/physpkg.F90` is a standard CAM file but some lines are changed so that `cam_gp.F90` is called once every 6 simulation hours.
