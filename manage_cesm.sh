#!/bin/bash

# ==============================================================================
# CESM 2.1.5 Case Manager (Full-Featured)
#
# Description:
#   This script creates, configures, builds, and runs CESM 2.1.5 cases.
#   For 'hybrid' runs, it mounts a complete, user-modified CAM source tree.
#
# --- USAGE EXAMPLES ---
#
# Create a standard 5-day low-resolution case:
# ./manage_cesm.sh create standard lowres 5 days /path/to/nl.txt
#
# Create a hybrid 1-year high-resolution case with monthly restarts:
# ./manage_cesm.sh create hybrid highres 1 year /path/to/nl.txt 1 month
#
# Run an existing case:
# ./manage_cesm.sh run hybrid F2000climo_1year_highres_nl_rest1month_hybrid
#
# ==============================================================================

set -e # Exit immediately on error

# --- ⚙️ CONFIGURATION ---
# --- Host Paths ---
HOST_BASE_DIR="/data/ucakjcb"
HOST_CASES_DIR="${HOST_BASE_DIR}/cases"
HOST_ARCHIVES_DIR="${HOST_BASE_DIR}/archives"
HOST_INPUT_DIR="${HOST_BASE_DIR}/CAM_input_files"

# --- Hybrid Mode Paths (ONLY used for 'hybrid' runs) ---
# This must be a COMPLETE copy of the CAM component, with your modifications.
HOST_CUSTOM_CAM_DIR="${HOST_BASE_DIR}/CAM_hybrid/cam"
# Directory on the HOST containing your PyTorch model files (e.g., .pt).
HOST_MODELS_DIR="${HOST_BASE_DIR}/models"

# --- Container Configuration ---
CONTAINER_IMAGE="cesm_ftorch:latest"
CONTAINER_CASES_DIR="/cases"
CONTAINER_ARCHIVE_DIR="/root/cesm/archive"
CONTAINER_MODELS_DIR="/models"
# This is the target directory inside the container that will be replaced.
CONTAINER_CAM_TARGET_DIR="/opt/cesm/components/cam"
CONTAINER_INPUT_DIR="/root/cesm/inputdata"


# --- Script Logic ---
usage() {
    echo "Usage: $0 <mode> <type> [options...]"
    echo
    echo "Modes:"
    echo "  create   Create and configure a new CESM case."
    echo "  run      Build and submit an existing CESM case."
    echo
    echo "Type:"
    echo "  standard   Standard CESM run (CPU)."
    echo "  hybrid     Custom CAM/FTorch run (mounts a full custom CAM directory)."
    echo
    echo "--- CREATE Usage ---"
    echo "$0 create <type> <resolution> <sim_length> <sim_units> <namelist_file> [<rest_freq> <rest_unit>]"
    echo
    echo "--- RUN Usage ---"
    echo "$0 run <type> <case_name>"
    exit 1
}

# --- Argument Parsing ---
if [ "$#" -eq 0 ]; then usage; fi
if [[ "$1" == "-h" ]]; then usage; fi

MODE=$1
RUN_TYPE=$2
shift 2

if [[ -z "$MODE" || -z "$RUN_TYPE" ]]; then usage; fi
case "$RUN_TYPE" in
    standard|hybrid) ;;
    *) echo "Error: Invalid type '${RUN_TYPE}'." >&2; usage ;;
esac
if ! command -v podman &> /dev/null; then
    echo "Error: 'podman' command not found." >&2; exit 1
fi

# --- Assemble Podman Command ---
DATETIME=$(date +'%Y%m%d%H%M')
podman_cmd=()
hybrid_flags=()

if [[ "$RUN_TYPE" == "hybrid" ]]; then
    echo "🧬 Hybrid mode selected. Mounting full custom CAM source tree."
    if [[ ! -d "${HOST_CUSTOM_CAM_DIR}" ]]; then
        echo "❌ Error: Custom CAM directory not found at '${HOST_CUSTOM_CAM_DIR}/'" >&2
        echo "   Please create a complete copy of the CAM component and add your modifications." >&2
        exit 1
    fi
    if [[ ! -d "${HOST_MODELS_DIR}" ]]; then
        echo "❌ Error: Models directory not found at '${HOST_MODELS_DIR}/'" >&2; exit 1
    fi
    hybrid_flags=(
        --device nvidia.com/gpu=all
        -v "${HOST_CUSTOM_CAM_DIR}:${CONTAINER_CAM_TARGET_DIR}:Z"
        -v "${HOST_MODELS_DIR}:${CONTAINER_MODELS_DIR}:ro,Z"
    )
fi

# --- Mode-Specific Logic ---
case "$MODE" in
    create)
        # --- Argument Parsing for Create ---
        if [[ "$#" -lt 4 ]]; then echo "Error: Missing arguments for 'create' mode" >&2; usage; fi
        RESOLUTION_KEY=$1; SIM_LENGTH=$2; SIM_UNITS=$3; NAMELIST_FILE=$4
        NAMELIST_FILENAME=$(basename "${NAMELIST_FILE}")
        if [[ "${NAMELIST_FILE}" != /* ]]; then NAMELIST_FILE="${PWD}/${NAMELIST_FILE}"; fi
        if [[ ! -f "${NAMELIST_FILE}" ]]; then echo "Error: Namelist file not found at '${NAMELIST_FILE}'" >&2; exit 1; fi
        case "$RESOLUTION_KEY" in
            "lowres")  RES_ARG="f19_f19_mg17"; RES_NAME="lowres" ;;
            "highres") RES_ARG="f09_f09_mg17"; RES_NAME="highres" ;;
            *) echo "Error: Invalid resolution '${RESOLUTION_KEY}'" >&2; exit 1 ;;
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
        if [ -d "${HOST_CASES_DIR}/${CASE_NAME}" ]; then echo "Error: Case directory already exists: ${HOST_CASES_DIR}/${CASE_NAME}" >&2; exit 1; fi

        CONTAINER_NAME="cesm-create-${CASE_NAME}-${DATETIME}"
        
        podman_cmd=(
            podman run -i --rm
            --name "${CONTAINER_NAME}"
            -v "${HOST_CASES_DIR}:${CONTAINER_CASES_DIR}:Z"
            -v "${NAMELIST_FILE}:${CONTAINER_CASES_DIR}/${NAMELIST_FILENAME}:ro,Z"
            "${hybrid_flags[@]}"
            "${CONTAINER_IMAGE}" /bin/bash
        )

        echo "🚀 Launching container to create case '${CASE_NAME}'..."
        
        "${podman_cmd[@]}" <<EOF
set -e
echo "--- [1/4] Inside container: Creating new case: ${CASE_NAME} ---"
cd /opt/cesm/cime/scripts
./create_newcase --case /cases/${CASE_NAME} --compset F2000climo --res ${RES_ARG}

cd /cases/${CASE_NAME}
echo ">>> Configuring namelist..."
cat "../${NAMELIST_FILENAME}" >> user_nl_cam
echo '' >> user_nl_cam

echo "--- [2/4] Running case.setup and configuring XML variables ---"
./xmlchange STOP_N=${SIM_LENGTH},STOP_OPTION=n${SIM_UNITS}
${XML_REST_COMMAND}
./case.setup

if [[ "${RUN_TYPE}" == "hybrid" ]]; then
    echo "--- [3/4] Directly injecting COMPILER and LINKER flags into Macros.make for HYBRID run ---"
    echo "" >> Macros.make
    echo "# --- User-defined flags for FTorch (injected by script) ---" >> Macros.make
    echo "FFLAGS += -I/usr/local/ftorch/include/ftorch" >> Macros.make
    echo "LDFLAGS += -L/opt/FTorch/build -lftorch" >> Macros.make
fi

echo "--- [4/4] Case '${CASE_NAME}' created and configured successfully. ---"
EOF
        echo "✅ Case workflow for '${CASE_NAME}' completed successfully."
        ;;

    run)
        CASE_NAME=$1
        HOST_CASE_PATH="${HOST_CASES_DIR}/${CASE_NAME}"
        if [ ! -d "${HOST_CASE_PATH}" ]; then
            echo "Error: Case directory not found at '${HOST_CASE_PATH}'" >&2;
            exit 1
        fi
        
        CONTAINER_NAME="cesm-run-${CASE_NAME}-${DATETIME}"
        HOST_ARCHIVE_PATH="${HOST_ARCHIVES_DIR}/${CASE_NAME}-${DATETIME}"
        mkdir -p "${HOST_ARCHIVE_PATH}"

        # Start with the base podman command array
        podman_cmd=(
            podman run -dit
            --name "${CONTAINER_NAME}"
            -v "${HOST_CASES_DIR}:${CONTAINER_CASES_DIR}:Z"
            -v "${HOST_ARCHIVE_PATH}:${CONTAINER_ARCHIVE_DIR}/${CASE_NAME}:Z"
        )

        # Conditionally add the low-resolution input data volume mount
        if [[ ${CASE_NAME} == *"lowres"* ]]; then
            echo "INFO: lowres case detected, adding local input data volume."
            podman_cmd+=(-v "${HOST_INPUT_DIR}/lowres/:${CONTAINER_INPUT_DIR}:Z")
        fi

        # Add the remaining flags and arguments
        podman_cmd+=(
            "${hybrid_flags[@]}"
        )

        COMMAND_TO_RUN="set -e; \
            echo '--- [1/3] Inside container: Running case: ${CASE_NAME} ---'; \
            cd /cases/${CASE_NAME}; \
            echo '--- [2/3] Building the case ---'; \
            ./case.build; \
            echo '--- [3/3] Submitting the run ---'; \
            ./case.submit; \
            echo '--- ✅ Run submitted successfully. ---'"

        podman_cmd+=("${CONTAINER_IMAGE}" /bin/bash -c "${COMMAND_TO_RUN}")

        echo "🚀 Launching container in detached mode to run case '${CASE_NAME}'..."
        container_id=$("${podman_cmd[@]}")

        HOST_LOG_FILE="${HOST_ARCHIVE_PATH}/podman_container.log"
        echo "✅ Container '${CONTAINER_NAME}' started successfully with ID: ${container_id:0:12}"
        echo
        echo "--- Next Steps ---"
        echo "  - The container is running in the background. Your terminal is free."
        echo "  - To monitor logs in real-time:"
        echo "    podman logs -f ${CONTAINER_NAME}"
        echo
        echo "  - A copy of the full output will be saved to:"
        echo "    ${HOST_LOG_FILE}"
        echo
        echo "  - The container will stop on its own but will NOT be removed."
        echo "  - To clean up the container after the run is finished:"
        echo "    podman rm ${CONTAINER_NAME}"
        
        podman logs -f "${CONTAINER_NAME}" > "${HOST_LOG_FILE}" 2>&1 &
        ;;
    *)
        usage
        ;;
esac

