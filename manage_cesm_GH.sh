#!/bin/bash

# ==============================================================================
# CESM 2.1.5 Case Manager for Isambard-AI (Multi-Node Ready)
# ==============================================================================

set -e # Exit immediately on error

# --- ⚙️ CONFIGURATION ---
HOST_BASE_DIR="${PROJECTDIR}/CAM-hybrid"
HOST_CASES_DIR="${HOST_BASE_DIR}/cases"
HOST_ARCHIVES_DIR="${HOST_BASE_DIR}/archives"
HOST_INPUT_DIR="${HOST_BASE_DIR}/CAM_input_files"
#HOST_SCRATCH_DIR="${HOST_BASE_DIR}/scratch"
# Pointing to the high-speed volatile Lustre file system
HOST_SCRATCH_DIR="${SCRATCHDIR}/CAM-hybrid/scratch"

# --- Hybrid Mode Paths ---
HOST_CUSTOM_CAM_DIR="${HOST_BASE_DIR}/CAM_hybrid/cam"
HOST_MODELS_DIR="${HOST_BASE_DIR}/models"

# --- Container Configuration ---
CONTAINER_IMAGE="docker.io/jamesbriant/cesm_ftorch"
AUTH="--authfile $HOME/my_docker_auth.json"
CONTAINER_CASES_DIR="/cases"
CONTAINER_ARCHIVE_DIR="/root/cesm/archive"
CONTAINER_MODELS_DIR="/models"
CONTAINER_CAM_TARGET_DIR="/opt/cesm/components/cam"
CONTAINER_INPUT_DIR="/root/cesm/inputdata"

# --- Helper for Timestamped Logging ---
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_err() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# --- Script Logic ---
usage() {
    echo "Usage: $0 <mode> <type> [options...]"
    echo "Modes:"
    echo "  create   Create and configure a new CESM case."
    echo "  run      Build case and generate a Slurm submission script."
    exit 1
}

if [ "$#" -eq 0 ] || [[ "$1" == "-h" ]]; then usage; fi

MODE=$1
RUN_TYPE=$2
shift 2

if [[ -z "$MODE" || -z "$RUN_TYPE" ]]; then usage; fi
case "$RUN_TYPE" in
    standard|hybrid) ;;
    *) log_err "Invalid type '${RUN_TYPE}'."; usage ;;
esac

DATETIME=$(date +'%Y%m%d%H%M')
hybrid_flags=()

if [[ "$RUN_TYPE" == "hybrid" ]]; then
    log_info "Hybrid mode selected. Mounting full custom CAM source tree."
    hybrid_flags=(
        --device nvidia.com/gpu=all
        -v "${HOST_CUSTOM_CAM_DIR}:${CONTAINER_CAM_TARGET_DIR}:Z"
        -v "${HOST_MODELS_DIR}:${CONTAINER_MODELS_DIR}:ro,Z"
    )
fi

# ==============================================================================
# MODE: CREATE
# ==============================================================================
if [[ "$MODE" == "create" ]]; then
    if [[ "$#" -lt 4 ]]; then log_err "Missing arguments for 'create'"; usage; fi
    RESOLUTION_KEY=$1; SIM_LENGTH=$2; SIM_UNITS=$3; NAMELIST_FILE=$4
    NAMELIST_FILENAME=$(basename "${NAMELIST_FILE}")
    if [[ "${NAMELIST_FILE}" != /* ]]; then NAMELIST_FILE="${PWD}/${NAMELIST_FILE}"; fi
    
    case "$RESOLUTION_KEY" in
        "lowres")  RES_ARG="f19_f19_mg17"; RES_NAME="lowres" ;;
        "highres") RES_ARG="f09_f09_mg17"; RES_NAME="highres" ;;
        *) log_err "Invalid resolution '${RESOLUTION_KEY}'"; exit 1 ;;
    esac
    
    XML_REST_COMMAND=""
    REST_INFO=""
    if [ "$#" -eq 6 ]; then
        REST_FREQ=$5; REST_UNIT=$6
        REST_INFO="_rest${REST_FREQ}${REST_UNIT}"
        XML_REST_COMMAND="./xmlchange REST_N=${REST_FREQ},REST_OPTION=n${REST_UNIT};"
    fi

    CASE_NAME="F2000climo_${SIM_LENGTH}${SIM_UNITS}_${RES_NAME}_$(basename "${NAMELIST_FILE}" .txt)${REST_INFO}"
    if [[ "$RUN_TYPE" == "hybrid" ]]; then CASE_NAME="${CASE_NAME}_hybrid"; fi
    
    CONTAINER_NAME="cesm-create-${CASE_NAME}-${DATETIME}"
    
    log_info "Preparing to launch container to create case '${CASE_NAME}'..."
    log_info "Podman container name will be: ${CONTAINER_NAME}"
    
    podman-hpc run -i --rm --pull=never --gpu $AUTH \
        --name "${CONTAINER_NAME}" \
        -v "${HOST_CASES_DIR}:${CONTAINER_CASES_DIR}:Z" \
        -v "${NAMELIST_FILE}:${CONTAINER_CASES_DIR}/${NAMELIST_FILENAME}:ro,Z" \
        "${hybrid_flags[@]}" \
        "${CONTAINER_IMAGE}" /bin/bash <<EOF
set -e
echo "[CONTAINER] $(date +'%H:%M:%S') - Starting case creation script inside container."

echo "[CONTAINER] $(date +'%H:%M:%S') - Executing create_newcase..."
cd /opt/cesm/cime/scripts
./create_newcase --case /cases/${CASE_NAME} --compset F2000climo --res ${RES_ARG}

echo "[CONTAINER] $(date +'%H:%M:%S') - Configuring user_nl_cam..."
cd /cases/${CASE_NAME}
cat "../${NAMELIST_FILENAME}" >> user_nl_cam
echo '' >> user_nl_cam

echo "[CONTAINER] $(date +'%H:%M:%S') - Executing xmlchange and case.setup..."
./xmlchange STOP_N=${SIM_LENGTH},STOP_OPTION=n${SIM_UNITS}
${XML_REST_COMMAND}
./case.setup

if [[ "${RUN_TYPE}" == "hybrid" ]]; then
    echo "[CONTAINER] $(date +'%H:%M:%S') - Injecting FTorch flags into Macros.make..."
    echo "" >> Macros.make
    echo "FFLAGS += -I/usr/local/ftorch/include/ftorch" >> Macros.make
    echo "LDFLAGS += -L/opt/FTorch/build -lftorch" >> Macros.make
fi
echo "[CONTAINER] $(date +'%H:%M:%S') - Case creation script complete."
EOF
    log_info "Case '${CASE_NAME}' created successfully on the host."

# ==============================================================================
# MODE: RUN (Builds the model and creates a Slurm script)
# ==============================================================================
elif [[ "$MODE" == "run" ]]; then
    CASE_NAME=$1
    NUM_NODES=${2:-1} # Defaults to 1 node if no argument is provided
    
    if [ -z "$CASE_NAME" ]; then log_err "Missing case name."; usage; fi
    
    HOST_CASE_PATH="${HOST_CASES_DIR}/${CASE_NAME}"
    HOST_ARCHIVE_PATH="${HOST_ARCHIVES_DIR}/${CASE_NAME}-${DATETIME}"
    
    log_info "Validating case path: ${HOST_CASE_PATH}"
    if [ ! -d "${HOST_CASE_PATH}" ]; then
        log_err "Case directory not found. Did you run 'create' first?"
        exit 1
    fi

    mkdir -p "${HOST_ARCHIVE_PATH}"
    log_info "Archive directory created at: ${HOST_ARCHIVE_PATH}"

    # --- Phase 1: Build the Executable & Download Data ---
    log_info "Phase 1: Building the case and downloading data inside container..."
    mkdir -p "${HOST_SCRATCH_DIR}" # Ensure the host directory exists
    
    # We must mount the input directory during Phase 1 so the downloaded data is saved to the host!
    INPUT_MOUNT=""
    if [[ ${CASE_NAME} == *"lowres"* ]]; then
        INPUT_MOUNT="-v ${HOST_INPUT_DIR}/lowres/:${CONTAINER_INPUT_DIR}:Z"
        mkdir -p "${HOST_INPUT_DIR}/lowres/"
    fi

    podman-hpc run -i --rm --pull=never --gpu $AUTH \
        -v "${HOST_CASES_DIR}:${CONTAINER_CASES_DIR}:Z" \
        -v "${HOST_SCRATCH_DIR}:/root/cesm/scratch:Z" \
        ${INPUT_MOUNT} \
        "${hybrid_flags[@]}" \
        "${CONTAINER_IMAGE}" /bin/bash <<EOF
set -e
echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Navigating to case directory..."
cd /cases/${CASE_NAME}

echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Starting ./case.build..."
./case.build

echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Downloading missing input data from SVN/FTP..."
./check_input_data --download

echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Build complete. Running ./preview_namelists..."
./preview_namelists

echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Phase 1 complete."
EOF
    log_info "Phase 1: Case built and data downloaded successfully."

# --- Phase 2: Prepare Volumes for Slurm ---
    log_info "Phase 2: Preparing container volumes for Slurm execution..."
    VOLUMES="-v ${HOST_CASES_DIR}:${CONTAINER_CASES_DIR}:Z -v ${HOST_ARCHIVE_PATH}:${CONTAINER_ARCHIVE_DIR}/${CASE_NAME}:Z -v ${HOST_SCRATCH_DIR}:/root/cesm/scratch:Z"
    
    if [[ ${CASE_NAME} == *"lowres"* ]]; then
        VOLUMES="${VOLUMES} -v ${HOST_INPUT_DIR}/lowres/:${CONTAINER_INPUT_DIR}:Z"
    fi
    if [[ "$RUN_TYPE" == "hybrid" ]]; then
        VOLUMES="${VOLUMES} -v ${HOST_CUSTOM_CAM_DIR}:${CONTAINER_CAM_TARGET_DIR}:Z -v ${HOST_MODELS_DIR}:${CONTAINER_MODELS_DIR}:ro,Z"
    fi

    # --- Phase 3: Determine Compute Requirements ---
    if [[ "$RUN_TYPE" == "hybrid" ]]; then
        GPU_STR="#SBATCH --gpus-per-node=4"
    else
        GPU_STR="" # No GPUs requested for CPU-only runs
    fi

    # --- Phase 4: Generate the Slurm Script ---
    SLURM_FILE="submit_${CASE_NAME}.slurm"
    log_info "Phase 3: Generating Slurm submission script -> ${SLURM_FILE}"
    
    cat <<EOF > "${SLURM_FILE}"
#!/bin/bash
#SBATCH --job-name=${CASE_NAME}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=4
${GPU_STR}
#SBATCH --time=06:00:00
#SBATCH --output=${HOST_ARCHIVE_PATH}/slurm-%j.out

echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Job \${SLURM_JOB_NAME} (ID: \${SLURM_JOB_ID}) started."
echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Allocated Nodes: \${SLURM_JOB_NODELIST}"

# Create the missing timing directories required by CESM
mkdir -p ${HOST_SCRATCH_DIR}/${CASE_NAME}/run/timing/checkpoints

# Executing across the high-speed fabric
echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Launching podman-hpc via srun..."

srun --mpi=pmi2 \\
    podman-hpc run --rm --pull=never --openmpi-pmi2 --gpu $AUTH \\
    ${VOLUMES} \\
    -w /root/cesm/scratch/${CASE_NAME}/run \\
    "${CONTAINER_IMAGE}" \\
    /root/cesm/scratch/${CASE_NAME}/bld/cesm.exe

EXIT_CODE=\$?

# Run the short-term archiver to rescue data from the volatile $SCRATCHDIR
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SLURM] $(date +'%Y-%m-%d %H:%M:%S') - Simulation successful. Archiving data back to /projects..."
    podman-hpc run --rm --pull=never $AUTH \
        ${VOLUMES} \
        -w /cases/${CASE_NAME} \
        "${CONTAINER_IMAGE}" \
        ./case.st_archive
fi

echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - srun completed with exit code: \${EXIT_CODE}"
echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Check outputs in ${HOST_ARCHIVE_PATH}"
exit \${EXIT_CODE}
EOF

    chmod +x "${SLURM_FILE}"
    log_info "✅ Success! To submit your simulation to the scheduler, run:"
    echo "   sbatch ${SLURM_FILE}"
fi
