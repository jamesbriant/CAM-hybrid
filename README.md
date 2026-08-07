# CAM-hybrid

This repository contains the codebase and scripts to run standard and hybrid (machine learning enhanced) versions of the Community Earth System Model (CESM) on the Isambard-AI supercomputer.

See the Medium article by Scotty for an outline of this project's workflow:
https://medium.com/@twins.corgi.0a/hybrid-ai-hpc-workflows-integrating-pytorch-with-cesm-on-nvidia-gh200-94a0bc34ffd9

## Codebase Structure

- `manage_cesm.sh`: The orchestration script for case creation, compilation, and Slurm job generation. 
- `manage_cesm_array.sh`: Same as `manage_cesm.sh` but for array jobs (e.g., sequentially running 1-year simulations).
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

See [GP Training](https://github.com/jamesbriant/CESM-GP).

Create a new directory and move the trained models to `models/your_trained_model.pt`.

Ensure lines 19-20 of `CAM_hybrid/cam/src/physics/cam/cam_gp.F90` are updated to point to the correct model path.

## Running the Model

We use containers (Podman-HPC) and a central script (`manage_cesm.sh`) to streamline execution on Isambard-AI.

### Step 1: Build Containers

**⚡ Quick Start:** You can skip the build process entirely! The containers pre-built for Isambard-AI and hosted on Docker Hub.
*[docker.io/jamesbriant/cesm_ftorch](https://hub.docker.com/repository/docker/jamesbriant/cesm_ftorch/general)*

1. You need an account with hub.docker.com - I used my GitHub account to create an account.
2. Create a personal access token in your account settings. This is needed for logging into you account via the CLI below...
3. Run the following on Isambard’s login node `podman-hpc login docker.io -u <YOUR_USERNAME> --authfile ~/my_docker_auth.json​`
4. Then run `podman-hpc pull docker.io/jamesbriant/cesm_ftorch​`

**Alternatively**, you can build the containers yourself. This is useful if you want to make changes to the CESM codebase or the FTorch inference code. There are 3 containers to build sequentially. These establish the CESM environment.

1. `cesm_base`
2. `cesm`
3. `cesm_ftorch`

This can be automated by submitting the `build.sh` script. Make sure the change the `USERNAME` variable to your Docker Hub username. The script will build and push the containers to your Docker Hub account. It take a long time, I have set 6 hours to build all three containers, I think this is sufficient. Submit the job on Isambard-AI with `sbatch build.sh`.

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
- `<sim_length>`: Length of the simulation without units (e.g., `30`).
- `<sim_units>`: Units of the simulation length (e.g., `days`, `months`, `years`).
- `<namelist_file>`: Path to the user namelist file (e.g., `cases/user_nl_cam.txt`).

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

### Step 3.1: Array Job Creation

Isambard has a max job time of 24 hours. If your simulation is longer than 24 hours, create an array job to run multiple sequential jobs. Use `./manage_cesm_array.sh create <type> <resolution> <sim_length> <sim_units> <namelist_file> <num_nodes>`.

### Step 4: Build and Generate Slurm Script

Once the case is created, compile the model and prepare it for the scheduler using `./manage_cesm.sh build <type> <case_name> [num_nodes]`.

- `<type>`: `standard` or `hybrid`.
- `<case_name>`: The name of the case generated in Step 3. This is the folder name in `cases/`. Use `<case_name>` to reference the case, not `cases/<case_name>`.
- `[num_nodes]`: Optional. Number of nodes to run on (remember each node has 4 GPUs). If not specified, defaults to 1 node (4 GPUs).

**Example: Standard mode**
```bash
./manage_cesm.sh build standard FHIST_30days_lowres_user_nl_cam-standard
```

**Example: Hybrid mode across 2 nodes**
```bash
./manage_cesm.sh build hybrid FHIST_30days_lowres_user_nl_cam_hybrid 2
```

This command will:
1. Compile the CESM executable inside the container.
2. Download any required input data (e.g., `lowres` boundary conditions).
3. Generate a submission script named `submit_<case_name>_<num_nodes>nodes.slurm`.

### Step 5: Launch

**BEFORE SUBMITTING:** Adjust the `SBATCH` directives in the generated Slurm script to match your desired walltime and array job configuration. 

The default walltime is set to **24 hours** and the default **array job is 21**. This means a new job will be submitted as soon as the previous job completes, 21 times. I haven't added this as a command line argument yet.

Submit the generated script to the Isambard-AI Slurm queue:

```bash
sbatch submit_<case_name>_<num_nodes>nodes.slurm
```

When the simulation completes successfully, the results will automatically be transferred from the fast scratch storage into the `archives/` directory.

## CAM Changes

Only two file changes are made to the CAM codebase.

`CAM_hybrid/cam/src/physics/cam/cam_gp.F90` is a new file containing the FTorch forward pass (GP prediction) implementation.

`CAM_hybrid/cam/src/physics/cam/physpkg.F90` is a standard CAM file but some lines are changed so that `cam_gp.F90` is called once every 6 simulation hours.

## Nuances of Running on Isambard-AI

Running complex containerized MPI workloads on an HPE Cray EX architecture like Isambard-AI requires a few specific configurations:

- **MPI and PMI Versions (pmi2 vs pmix):** Isambard-AI uses Slurm for job scheduling, which natively interfaces with MPI. We originally investigated using `pmix` and `pmix_v5`, which are standard on many modern HPC systems. However, within the `podman-hpc` container environment on Isambard-AI, we found that compiling with and specifying `--mpi=pmi2` in the `srun` command, combined with passing `--openmpi-pmi2` to the container process, proved to be the only successful combination.
- **Inter-Process Communication (`--ipc=host`):** By default, containers isolate the IPC namespace. For MPI tasks to communicate efficiently (especially when utilizing GPU acceleration across node boundaries), the container must share the host's IPC namespace. Passing `--ipc=host` to `podman-hpc` prevents shared memory allocation errors and deadlocks during multi-node runs.
- **Container Device Mounts:** When interacting with GPUs, `podman-hpc` requires the `--gpu` flag to securely mount the NVIDIA devices and CDI (Container Device Interface) configurations from the login and compute nodes seamlessly.
