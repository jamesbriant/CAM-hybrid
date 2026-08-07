#!/bin/bash

# ==============================================================================
# CESM Input Data Downloader
# ==============================================================================

set -e

if [ -z "$1" ] || [[ ! "$1" =~ ^(lowres|highres)$ ]]; then
    echo "Usage: $0 <lowres|highres>"
    exit 1
fi

RES_KEY=$1
if [ "$RES_KEY" == "lowres" ]; then
    RES_ARG="f19_f19_mg17"
else
    RES_ARG="f09_f09_mg17"
fi

# Paths based on the parent framework
PROJECTDIR="/projects/u6t"
HOST_INPUT_DIR="${PROJECTDIR}/CAM-hybrid-oxford/CAM_input_files/${RES_KEY}"
CONTAINER_IMAGE="docker.io/jamesbriant/cesm_ftorch"
AUTH="--authfile $HOME/my_docker_auth.json"

mkdir -p "${HOST_INPUT_DIR}"
echo "Downloading ${RES_KEY} (${RES_ARG}) data to ${HOST_INPUT_DIR}..."

# Run a temporary container to create a dummy case and download the data
podman-hpc run -i --rm --pull=never $AUTH \
    -v "${HOST_INPUT_DIR}:/root/cesm/inputdata:Z" \
    "${CONTAINER_IMAGE}" /bin/bash <<EOF
set -e
echo "-> Creating dummy case for ${RES_KEY}..."
cd /opt/cesm/cime/scripts
./create_newcase --case /tmp/dummy_case_${RES_KEY} --compset FHIST --res ${RES_ARG} --run-unsupported

echo "-> Setting up case..."
cd /tmp/dummy_case_${RES_KEY}
./xmlchange RUN_TYPE=startup,RUN_STARTDATE=1984-01-01

./case.setup

echo "-> Generating namelists to determine required data..."
./preview_namelists

echo "-> Downloading data via SVN/FTP (this may take a while)..."
./check_input_data --download
EOF

echo "✅ Download complete! Data is available in: ${HOST_INPUT_DIR}"