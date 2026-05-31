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
HOST_CAM_SRC_DIR="${HOST_CUSTOM_CAM_DIR}/src/physics/cam"

# --- Container Configuration ---
CONTAINER_IMAGE="docker.io/jamesbriant/cesm_ftorch"
AUTH="--authfile $HOME/my_docker_auth.json"
CONTAINER_CASES_DIR="/cases"
CONTAINER_ARCHIVE_DIR="/root/cesm/archive"
CONTAINER_MODELS_DIR="/models"
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
    echo "  build    Build case and generate a Slurm submission script."
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
    log_info "Hybrid mode selected. Staging FTorch CAM files into SourceMods/src.cam."
    hybrid_flags=(
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

    NAMELIST_BASENAME=$(basename "${NAMELIST_FILE}" .txt)
    # Strip "-hybrid" or "_hybrid" from the namelist name so we don't duplicate it
    NAMELIST_BASENAME=${NAMELIST_BASENAME%-hybrid}
    NAMELIST_BASENAME=${NAMELIST_BASENAME%_hybrid}

    CASE_NAME="F2000climo_${SIM_LENGTH}${SIM_UNITS}_${RES_NAME}_${NAMELIST_BASENAME}${REST_INFO}"
    if [[ "$RUN_TYPE" == "hybrid" ]]; then CASE_NAME="${CASE_NAME}_hybrid"; fi
    
    CONTAINER_NAME="cesm-create-${CASE_NAME}-${DATETIME}"
    
    log_info "Preparing to launch container to create case '${CASE_NAME}'..."
    log_info "Podman container name will be: ${CONTAINER_NAME}"
    
    podman-hpc run -i --rm --pull=never --gpu $AUTH \
        --name "${CONTAINER_NAME}" \
        -v "${HOST_CASES_DIR}:${CONTAINER_CASES_DIR}:Z" \
        -v "${HOST_CAM_SRC_DIR}:/cam_src:ro,Z" \
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

if [[ "${RUN_TYPE}" == "hybrid" ]]; then
    echo "[CONTAINER] $(date +'%H:%M:%S') - Staging FTorch CAM files into SourceMods/src.cam..."
    mkdir -p SourceMods/src.cam
    cp /cam_src/cam_gp.F90 SourceMods/src.cam/
    cp /cam_src/physpkg.F90 SourceMods/src.cam/
fi

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
# MODE: BUILD (Builds the model and creates a Slurm script)
# ==============================================================================
elif [[ "$MODE" == "build" ]]; then
    CASE_NAME=$1
    NUM_NODES=${2:-1} # Defaults to 1 node if no argument is provided
    
    if [ -z "$CASE_NAME" ]; then log_err "Missing case name."; usage; fi
    
    # Extract SIM_LENGTH and SIM_UNITS from CASE_NAME
    # Example format: F2000climo_30days_lowres_user_nl...
    SIM_LENGTH=1
    SIM_UNITS="unknown"
    if [[ "${CASE_NAME}" =~ ^F2000climo_([0-9]+)([a-zA-Z]+)_ ]]; then
        SIM_LENGTH="${BASH_REMATCH[1]}"
        SIM_UNITS="${BASH_REMATCH[2]}"
    fi
    
    HOST_CASE_PATH="${HOST_CASES_DIR}/${CASE_NAME}"
    
    # Simplify archive name: Remove F2000climo_ and user_nl* combinations
    SHORT_NAME=$(echo "${CASE_NAME}" | sed -E 's/^F2000climo_//' | sed -E 's/_user_nl_[a-zA-Z0-9_\-]+//')
    if [[ "$RUN_TYPE" == "hybrid" ]]; then
        SHORT_NAME="${SHORT_NAME}_hybrid"
    fi
    HOST_ARCHIVE_PATH="${HOST_ARCHIVES_DIR}/${SHORT_NAME}_${NUM_NODES}nodes-${DATETIME}"
    
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
        -v "${HOST_CAM_SRC_DIR}:/cam_src:ro,Z" \
        -v "${HOST_SCRATCH_DIR}:/root/cesm/scratch:Z" \
        ${INPUT_MOUNT} \
        "${hybrid_flags[@]}" \
        "${CONTAINER_IMAGE}" /bin/bash <<EOF
set -e
echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Navigating to case directory..."
cd /cases/${CASE_NAME}

echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Re-tuning NTASKS if necessary..."
TOTAL_RANKS=$(( ${NUM_NODES:-1} * 4 ))
cur=\$(./xmlquery --value NTASKS | head -1)
if [ "\$cur" != "\${TOTAL_RANKS}" ]; then
    echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Changing NTASKS \$cur -> \${TOTAL_RANKS}"
    ./xmlchange NTASKS=\${TOTAL_RANKS}
    ./case.setup --reset
    ./case.build --clean-all || true
fi

if [[ "${RUN_TYPE}" == "hybrid" ]]; then
    echo "[BUILD_CONTAINER] \$(date +'%H:%M:%S') - Refreshing SourceMods/src.cam from host FTorch files..."
    mkdir -p SourceMods/src.cam
    cp /cam_src/cam_gp.F90 SourceMods/src.cam/
    cp /cam_src/physpkg.F90 SourceMods/src.cam/
fi

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
        VOLUMES="${VOLUMES} -v ${HOST_CAM_SRC_DIR}:/cam_src:ro,Z -v ${HOST_MODELS_DIR}:${CONTAINER_MODELS_DIR}:ro,Z"
    fi

    # --- Phase 3: Determine Compute Requirements ---
    TOTAL_RANKS=$(( NUM_NODES * 4 ))
    if [[ "$RUN_TYPE" == "hybrid" ]]; then
        GPU_STR="#SBATCH --gpus-per-node=4"
        SRUN_GPU_STR="--gpus-per-node=4"
    else
        GPU_STR="" # No GPUs requested for CPU-only runs
        SRUN_GPU_STR=""
    fi

    # --- Phase 4: Generate the Slurm Script ---
    SLURM_FILE="submit_${CASE_NAME}_${NUM_NODES}nodes.slurm"
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

# Record job start time for timing metadata
JOB_START_TIME=\$(date +%s)
JOB_START_DATE=\$(date +'%Y-%m-%d %H:%M:%S')

# Create the missing timing directories required by CESM
mkdir -p ${HOST_SCRATCH_DIR}/${CASE_NAME}/run/timing/checkpoints

# Executing across the high-speed fabric
echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Launching podman-hpc via srun..."

srun --mpi=pmi2 ${SRUN_GPU_STR} \\
    podman-hpc run --rm --pull=never --openmpi-pmi2 --gpu $AUTH --ipc=host \\
    ${VOLUMES} \\
    -w /root/cesm/scratch/${CASE_NAME}/run \\
    "${CONTAINER_IMAGE}" \\
    /root/cesm/scratch/${CASE_NAME}/bld/cesm.exe

EXIT_CODE=\$?

# Run the short-term archiver to rescue data from the volatile $SCRATCHDIR
if [ \$EXIT_CODE -eq 0 ]; then
    echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Simulation successful. Archiving data back to /projects..."
    podman-hpc run --rm --pull=never \$AUTH \\
        ${VOLUMES} \\
        -w /cases/${CASE_NAME} \\
        "${CONTAINER_IMAGE}" \\
        ./case.st_archive

    # Generate metadata metrics
    JOB_END_TIME=\$(date +%s)
    JOB_END_DATE=\$(date +'%Y-%m-%d %H:%M:%S')
    DURATION=\$(( JOB_END_TIME - JOB_START_TIME ))

    # Convert simulation length to standard "months" equivalent metric for calculation
    # Approximation: 1 year = 12 months, 1 month = 1 month, 1 day = 1/30 month
    SIM_MONTHS=0
    if [[ "${SIM_UNITS}" == *"year"* ]]; then
        SIM_MONTHS=\$(echo "${SIM_LENGTH} * 12" | bc -l)
    elif [[ "${SIM_UNITS}" == *"month"* ]]; then
        SIM_MONTHS=${SIM_LENGTH}
    elif [[ "${SIM_UNITS}" == *"day"* ]]; then
        SIM_MONTHS=\$(echo "scale=4; ${SIM_LENGTH} / 30" | bc -l)
    fi

    # Calculate metrics (avoiding div by zero)
    CALC_PER_MONTH="N/A"
    CALC_PER_NODE="N/A"
    CALC_PER_MONTH_PER_NODE="N/A"
    
    if (( \$(echo "\$SIM_MONTHS > 0" | bc -l) )); then
        CALC_PER_MONTH=\$(echo "scale=2; \$DURATION / \$SIM_MONTHS" | bc -l)
    fi
    
    if [ ${NUM_NODES} -gt 0 ]; then
        CALC_PER_NODE=\$(echo "scale=2; \$DURATION / ${NUM_NODES}" | bc -l)
        if (( \$(echo "\$SIM_MONTHS > 0" | bc -l) )); then
            CALC_PER_MONTH_PER_NODE=\$(echo "scale=4; (\$DURATION / 3600 / \$SIM_MONTHS) * ${NUM_NODES}" | bc -l)
        fi
    fi

    METADATA_FILE="${HOST_ARCHIVE_PATH}/timing_metrics.txt"
    echo "Start time: \$JOB_START_DATE" > "\$METADATA_FILE"
    echo "End time: \$JOB_END_DATE" >> "\$METADATA_FILE"
    echo "Duration (seconds): \$DURATION" >> "\$METADATA_FILE"
    echo "Simulation length in months: \$SIM_MONTHS" >> "\$METADATA_FILE"
    echo "Time spent calculating per month (seconds): \$CALC_PER_MONTH" >> "\$METADATA_FILE"
    echo "Number of nodes used: ${NUM_NODES}" >> "\$METADATA_FILE"
    echo "Time spent per node (seconds): \$CALC_PER_NODE" >> "\$METADATA_FILE"
    echo "Node-hours per month constraint: \$CALC_PER_MONTH_PER_NODE" >> "\$METADATA_FILE"
    
    echo "[SLURM] Timing metadata saved to \$METADATA_FILE"
fi

echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - srun completed with exit code: \${EXIT_CODE}"
echo "[SLURM] \$(date +'%Y-%m-%d %H:%M:%S') - Check outputs in ${HOST_ARCHIVE_PATH}"
exit \${EXIT_CODE}
EOF

    chmod +x "${SLURM_FILE}"
    log_info "✅ Success! To submit your simulation to the scheduler, run:"
    echo "   sbatch ${SLURM_FILE}"
fi
