#!/usr/bin/env bash
#
# Usage:
#   ./run_three_stage_pipeline.sh
#       Run the full pipeline using SOURCE_SCAN_IDS defined in this script.
#
#   ./run_three_stage_pipeline.sh <scan_id>
#       Run the pipeline for a single scan ID (plain string, no quotes needed):
#           ./run_three_stage_pipeline.sh 800c7738-239b-46db-9612-328411369a9d
#
#   ./run_three_stage_pipeline.sh '<list>'
#       Run the pipeline for multiple scan IDs supplied as a list.
#       Single quotes are required to prevent shell interpretation of brackets:
#           ./run_three_stage_pipeline.sh '["id-1", "id-2"]'
#
set -euo pipefail       # Make script fail immediately if any step fails

# ============================
# User-configurable variables
# ============================
#
# ---------------------------------------------------------------------------
# 1. SOURCE_SCAN_IDS, HOSPITAL - these may change with each run of the script
# ---------------------------------------------------------------------------
#  
# Source_scan_ids value to process - unless given as cmd line argument.
# These will be written into JSON config files for all steps. Mind that
# when specific IDs are given, they can only refer to a single hospital specified
# below - a list of hospitals is not allowed.
# Accepted forms for scan_ids:
# - null
# - JSON list of scan IDs, for example: ["scan-id-1", "scan-id-2"]

SOURCE_SCAN_IDS='null'          # Mind the SINGLE QUOTES around the given value
#SOURCE_SCAN_IDS='["800c7738-239b-46db-9612-328411369a9d"]'     # Example for a single scan ID

# Hospital value that will be written into JSON configs for each step.
# Accepted forms:
# - single hospital string: Burnaby
# - JSON list of strings: ["Burnaby", "RCH"]
#
HOSPITAL='["Abbotsford"]'       # Mind the SINGLE QUOTES here

# ------------------------------------------------------
# 2. HOST - set the host machine were processing is done
# ------------------------------------------------------

# Host key under the "hosts" section in each JSON. If it is neither
# "fir" nor "ub2" - use "other"
HOST="fir"

# ---------------------------------------------------------------
# 3. Settings below will rarely need to be changed. They define
#    paths INSIDE the container, and JSON configuration file names
# ---------------------------------------------------------------

# Absolute path to the preprocessing project folder.
SOURCE_FOLDER="/app"

# Virtual environment directory to activate (must contain bin/activate).
VENV="/opt/venv"

# Root paths to apply to host-specific keys in the 3 JSON files.
# Under each path, their will be subfolders for each hospital
ORG_EDF_ROOT="/org_edf_root"                # Original raw EDFs
SEGMENTED_EDF_ROOT="/segmented_edf_root"    # Good segments EDFs, filtered/downsampled
PYPREPED_FIF_ROOT="/pypreped_fif_root"      # Records in .fif format, after pyprep and ICA
BEAMFORMED_ROOT="/beamformed_root"          # Source reconstructed results (.fif for forward sol, .hdf5 for time courses)

# Exactly 3 JSON config files for the preporcessing steps, in this order:
# 1) run_filtering_segmentation's config JSON
# 2) run_pyprep_ica's JSON
# 3) run_src_reconstr's JSON
INPUT_JSONS=(
  "preproc_conf.json"
  "pyprep_ica_conf.json"
  "src_reconstr_conf.json"
)

# ==== end of settings =======

# ----------------------------
# Actions
# ----------------------------

# Optional CLI override:
#   ./run_three_stage_pipeline.sh <scan_id>
# If provided, force SOURCE_SCAN_IDS to a one-item JSON list with this ID as string.
if [[ $# -gt 1 ]]; then
    echo "Error: At most one optional argument is supported: <scan_id>." >&2
    exit 1
fi

if [[ $# -eq 1 ]]; then
    SCAN_ID_ARG="$1"
    if [[ "${SCAN_ID_ARG}" == \[* ]]; then
        # Argument starts with '[': treat as a pre-formatted JSON list and use directly.
        SOURCE_SCAN_IDS="${SCAN_ID_ARG}"
        echo "Using scan ID list from CLI argument: ${SCAN_ID_ARG}"
    else
        # Plain scan ID string: escape and wrap in a one-item JSON list.
        # The ${VAR//PATTERN/REPLACEMENT} bash construct is used to replace text patterns
        # within a variable value. The leading double backslash tells to replace all
        # occurences of the pattern, not just the first one
        SCAN_ID_JSON_ESCAPED="${SCAN_ID_ARG//\\/\\\\}"          # Escape backslashes - just in case
        SCAN_ID_JSON_ESCAPED="${SCAN_ID_JSON_ESCAPED//\"/\\\"}" # Escape quotes - just in case
        SOURCE_SCAN_IDS="[\"${SCAN_ID_JSON_ESCAPED}\"]"         # Put quoted scan ID in square brackets
        echo "Using single scan ID from CLI argument: ${SCAN_ID_ARG}"
    fi
fi

cd "$SOURCE_FOLDER"

# Verify sibling folders next to SOURCE_FOLDER.
SOURCE_PARENT="$(dirname "$SOURCE_FOLDER")"
for required_dir in misc beam-python; do
    if [[ ! -d "$SOURCE_PARENT/$required_dir" ]]; then
        echo "Error: Required additional source folder is missing: $SOURCE_PARENT/$required_dir" >&2
        exit 1
    fi
done

# Activating virtual environment...
if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
else
    echo "Error: VENV must be a virtualenv directory containing bin/activate: $VENV" >&2
  exit 1
fi

# Count JSONs in the list - should be exactly three
if [[ ${#INPUT_JSONS[@]} -ne 3 ]]; then
  echo "Error: INPUT_JSONS must contain exactly 3 file paths." >&2
  exit 1
fi

JSON1="${INPUT_JSONS[0]}"
JSON2="${INPUT_JSONS[1]}"
JSON3="${INPUT_JSONS[2]}"

echo "Updating JSON configuration files..."
# Syntax:
#   "-" - read standard input
#   'PY' - use everyting until PY literally without interpretation 
python3 - "$JSON1" "$JSON2" "$JSON3" "$HOST" "$HOSPITAL" "$SOURCE_SCAN_IDS" "$ORG_EDF_ROOT" "$SEGMENTED_EDF_ROOT" "$PYPREPED_FIF_ROOT" "$BEAMFORMED_ROOT" <<'PY'
# -------------------------------------------
# Use this python script for all JSON editing
# -------------------------------------------
import json
import pathlib
import sys

import commentjson as cjson

def parse_hospital(raw):
    text = raw.strip()
    try:
        value = cjson.loads(text)
    except Exception:
        value = text

    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError('HOSPITAL must be either a string or a list of strings.')


def parse_source_scan_ids(raw):
    text = raw.strip()
    if text.lower() == "null":
        return None

    try:
        value = cjson.loads(text)
    except Exception as exc:
        raise ValueError('SOURCE_SCAN_IDS must be null or a JSON list of strings.') from exc

    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError('SOURCE_SCAN_IDS must be null or a JSON list of strings.')


def load_cfg(path):
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with cfg_path.open('r', encoding='utf-8') as fp:
        return cjson.loads(fp.read())


def apply_common_fields(cfg, hospital_value, source_scan_ids_value):
    cfg['hospital'] = hospital_value
    cfg['source_scan_ids'] = source_scan_ids_value


def write_cfg(path, cfg):
    cfg_path = pathlib.Path(path)
    with cfg_path.open('w', encoding='utf-8') as fp:
        json.dump(cfg, fp, indent=4)
        fp.write('\n')


(
    json1_path,
    json2_path,
    json3_path,
    host,
    hospital_raw,
    source_scan_ids_raw,
    org_edf_root,
    segmented_edf_root,
    pypreped_fif_root,
    beamformed_root,
) = sys.argv[1:]

hospital_value = parse_hospital(hospital_raw)
source_scan_ids_value = parse_source_scan_ids(source_scan_ids_raw)

cfg1 = load_cfg(json1_path)
cfg2 = load_cfg(json2_path)
cfg3 = load_cfg(json3_path)

# Set hospital and scan IDs to all JSONs
apply_common_fields(cfg1, hospital_value, source_scan_ids_value)
apply_common_fields(cfg2, hospital_value, source_scan_ids_value)
apply_common_fields(cfg3, hospital_value, source_scan_ids_value)

# Set hospital and scan IDs to all JSONs
# This will fail if host is not listed in the "hosts" key
host1 = cfg1['hosts'][host]
host2 = cfg2['hosts'][host]
host3 = cfg3['hosts'][host]

# Requested substitutions:
# org_edf_root       -> json1.hosts.<host>.data_root
# segmented_edf_root -> json1.hosts.<host>.out_root and json2.hosts.<host>.data_root
# pypreped_fif_root  -> json2.hosts.<host>.out_root and json3.hosts.<host>.data_root
# beamformed_root    -> json3.hosts.<host>.out_root
host1['data_root'] = org_edf_root
host1['out_root'] = segmented_edf_root

host2['data_root'] = segmented_edf_root
host2['out_root'] = pypreped_fif_root

host3['data_root'] = pypreped_fif_root
host3['out_root'] = beamformed_root

write_cfg(json1_path, cfg1)
write_cfg(json2_path, cfg2)
write_cfg(json3_path, cfg3)

print('Updated:', json1_path)
print('Updated:', json2_path)
print('Updated:', json3_path)
PY

echo "Running filtering and segmentation..."
python3 run_filtering_segmentation.py

echo "Running PyPREP + ICA..."
python3 run_pyprep_ica.py

echo "Running source reconstruction..."
python3 run_src_reconstr.py

echo "Pipeline completed successfully."
