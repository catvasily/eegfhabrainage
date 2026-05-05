#!/usr/bin/env bash
#
# Bind folders and run EEG preprocessing steps with apptainer
#
# Usage:
#   bash run_apptainer.sh [<source-scan-IDs-list>]
# -----------------------------------------------------------

# ============================
# User-configurable variables
# ============================

# Set this to non-zero value to use apptainer in instance mode
# that is - with preloading image into the host OS system. It makes
# sense if this script will be run multiple times in a single session.
PRELOAD_CONTAINER=1

# The folder from which apptainer will be run. Working copies of JSON files
# preproc_conf.json, pyprep_ica_conf.json,src_reconstr_conf.json should be
# placed here
WORKDIR=/project/6019337/amoiseev/containers

# External folders that should be accessible from container:
RPP="/project/6019337"
ORG_EDF_ROOT="$RPP/databases/eeg_fha/release_001/edf_subset"
SEGMENTED_EDF_ROOT="$RPP/$USER/data/eegfhabrainage/processed"
PYPREPED_FIF_ROOT="$RPP/$USER/data/eegfhabrainage/after-prep-ica"
BEAMFORMED_ROOT="$RPP/$USER/data/eegfhabrainage/src-reconstr"

# Path to the folder that contains the standard 'fsaverage' subfolder
# with surfaces, atlases, etc.
FREESURFER_DIR="$RPP/amoiseev/data/mne_data/MNE-fsaverage-data"
#FREESURFER_DIR="$RPP/$USER/data/mne_data/MNE-fsaverage-data"

# The apptainer image to use:
IMAGE=eegfh_260503.sif

# The command to run by the apptainer
COMMAND=./run_three_stage_pipeline.sh

# Names of config files for each step
JSON1=preproc_conf.json
JSON2=pyprep_ica_conf.json
JSON3=src_reconstr_conf.json

# ==== end of settings =======

# ----------------------------
# Actions
# ----------------------------

cd $WORKDIR
module load apptainer

# Construct a full list of mappings to the paths inside the container
BIND="$ORG_EDF_ROOT:/org_edf_root,$SEGMENTED_EDF_ROOT:/segmented_edf_root,$PYPREPED_FIF_ROOT:/pypreped_fif_root,$BEAMFORMED_ROOT:/beamformed_root"
BIND=$BIND",$PWD/$JSON1:/app/$JSON1,$PWD/$JSON2:/app/$JSON2,$PWD/$JSON3:/app/$JSON3"
BIND=$BIND",$FREESURFER_DIR:$FREESURFER_DIR"	# ... add more like this if needed

INST=eeg_prep	# Name for the apptainer instance, if preloading is used. Choose as you want.

# Start apptainer from a folder containing executable script run_three_stage_pipeline.sh,
# and three writable configuration files: preproc_conf.json, pyprep_ica_conf.json, src_reconstr_conf.json
# If a command line argument is specified for this script, it is interpreted as a single scan ID to be
# processed (i.e.122ad573-f2e2-46b3-b2ce-266bb8ce605c) by the script set in COMMAND variable above

if [ "${PRELOAD_CONTAINER}" -ne 0 ]; then
    # Check if instance is already running
    if ! apptainer instance list | awk '{print $1}' | grep -qx "${INST}"; then
        apptainer instance start \
            --bind ${BIND} \
            "${IMAGE}" "${INST}"
    fi

    # Run command in instance
    apptainer exec instance://"${INST}" ${COMMAND} $1

else
    # Run normally without instance
    apptainer exec \
        --bind ${BIND} \
        "${IMAGE}" ${COMMAND} $1
fi

