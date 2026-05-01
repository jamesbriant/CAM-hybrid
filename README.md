# CAM-hybrid

This repository contains the codebase and scripts to run standard and hybrid (machine learning enhanced) versions of the Community Earth System Model (CESM) on the Isambard-AI supercomputer.

See the Medium article by Scotty for an outline of this project's workflow:
https://medium.com/@twins.corgi.0a/hybrid-ai-hpc-workflows-integrating-pytorch-with-cesm-on-nvidia-gh200-94a0bc34ffd9

## Codebase Structure

- `manage_cesm.sh`: The main wrapper script to orchestrate case creation, compilation, and Slurm job generation.
- `CAM_hybrid/`: Contains the custom CAM source code used to overwrite base CESM. Custom physical parameterizations (e.g., the neural network/GP inference code via FTorch in `cam_gp.F90`) live here.
- `cases/`: Directory for user namelists (`user_nl_cam*` etc).
- `models/`: Stores trained PyTorch models (`.pt`) for hybrid inferences. 
- `cesm/`, `cesm_base/`, `cesm_ftorch/`: Container build scripts and Dockerfiles used to create the required podman/docker environments.
- `archives/`: Automatically generated folder where successful case simulation outputs and check-point data are saved.

## Installation

```bash
git clone https://github.com/jamesbriant/CAM-hybrid.git
cd CAM-hybrid
```

## GP Training

Train the Gaussian process model before proceeding to running the hybrid model.

See [ENTER REPO URL HERE]

Create a new directory and move the trained models to `models/your_trained_model.pt`.

## Running the Model

We use containers (Podman-HPC) and a central script (`manage_cesm.sh`) to streamline execution on Isambard-AI.

### Step 1: Build Containers

**⚡ Quick Start:** You can skip the build process entirely! The final container is pre-built and hosted on Docker Hub. `manage_cesm.sh` is already configured to use it.
*Link: [docker.io/jamesbriant/cesm_ftorch](https://hub.docker.com/repository/docker/jamesbriant/cesm_ftorch/general)*

If you need to make changes and build them yourself, there are 3 containers to build sequentially. These establish the CESM environment.

1. `cesm_base`
2. `cesm`
3. `cesm_ftorch`

**Important build notes for Isambard-AI:**
- Each build stage needs to be run independently on the compute nodes.
- Make sure to use `podman-hpc` to run the build scripts on Isambard-AI. It automatically loads the relevant runtime header files to compile properly.
- Isambard's local squashed image framework can be difficult to chain builds with. The recommended workflow is: after each build, push the image to `hub.docker.com`, and then pull from that remote image in the next stage's Dockerfile.

### Step 2: Configure Paths

Open `manage_cesm.sh` and update the `HOST_BASE_DIR` variable to match the absolute path to this repository on your system. By default, it expects `$PROJECTDIR` and `$SCRATCHDIR` variables to be set.

### Step 3: Create a Case

Use `./manage_cesm.sh create <type> <resolution> <sim_length> <sim_units> <namelist_file>` to configure a new case.

- `<type>`: `standard` or `hybrid`.
- `<resolution>`: `lowres` (f19_f19) or `highres` (f09_f09).

**Example: Creating a standard CESM run (30 days)**
```bash
./manage_cesm.sh create standard lowres 30 days cases/user_nl_cam-standard.txt
```

**Example: Creating a hybrid CESM run (30 days)**
*Make sure your `.pt` files are properly placed in `models/`.*
```bash
./manage_cesm.sh create hybrid lowres 30 days cases/user_nl_cam-hybrid.txt
```

*Note: This generates the case folder in `cases/` and configures your namelist files. It will output a `<case_name>`.*

### Step 4: Build and Generate Slurm Script

Once the case is created, compile the model and prepare it for the scheduler using `./manage_cesm.sh run <type> <case_name> [num_nodes]`.

**Example: Standard mode**
```bash
./manage_cesm.sh run standard F2000climo_30days_lowres_user_nl_cam-standard
```

**Example: Hybrid mode across 2 nodes**
```bash
./manage_cesm.sh run hybrid F2000climo_30days_lowres_user_nl_cam-hybrid_hybrid 2
```

This command will:
1. Compile the CESM executable inside the container.
2. Download any required input data (e.g., `lowres` boundary conditions).
3. Generate a submission script named `submit_<case_name>.slurm`.

### Step 5: Launch

Submit the generated script to the Isambard-AI Slurm queue:

```bash
sbatch submit_<case_name>.slurm
```

When the simulation completes successfully, the results will automatically be transferred from the fast scratch storage into the `archives/` directory.

## CAM Changes

Only two file changes are made to the CAM codebase.

`CAM_hybrid/cam/src/physics/cam/cam_gp.F90` is a new file containing the FTorch forward pass (GP prediction) implementation.

`CAM_hybrid/cam/src/physics/cam/physpkg.F90` is a standard CAM file but some lines are changed so that `cam_gp.F90` is called once every 6 simulation hours.
