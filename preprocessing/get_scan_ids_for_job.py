'''
When running an array job on cluster. get a list of records to be processed
for specified job #
'''
import sys
import os
import os.path as path
import glob
import commentjson as cjson
from run_welch import get_data_folders

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

INPUT_JSON_FILE = "cwt_input.json"  # This script input parameters
OUTPUT_LIST_FILE = lambda job_id: f"scan_ids_{job_id}.txt"

def main():
    if len(sys.argv) == 1:   # No command line args
        ijob = 0
    else:
        ijob = int(sys.argv[1])

    # Parse input args
    with open(pathname(INPUT_JSON_FILE), 'r') as fp:
        args = cjson.loads(fp.read())

    N_ARRAY_JOBS = args['N_ARRAY_JOBS']     # Number of parallel jobs to run on cluster
    what = args['what']                     # 'sensors' or 'sources'
    hospital = args['hospital']             # Burnaby, Abbotsford, RCH, Surrey
    source_scan_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf)

    data_root, out_root, cluster_job = get_data_folders(args)
    input_dir = data_root + "/" + hospital

    if source_scan_ids is None:
        if what == 'sensors':
            # To get bare ID need to chop off "_raw.fif" at the end
            source_scan_ids = [path.basename(f)[:-8] for f in glob.glob(input_dir + '/*.fif')]
        else:
            # To get bare ID one needs to get folders with names that are 5 HEX numbers
            # separated by 4 dashes. Poor man's solution for it is just '*-*-*-*-*'
            source_scan_ids = [path.basename(f) for f in glob.glob(input_dir + '/*-*-*-*-*')]

    nfiles = len(source_scan_ids)
    files_per_job = nfiles // N_ARRAY_JOBS + 1
    istart = ijob * files_per_job

    if istart > nfiles - 1:
        print(f'No files to process in job #{ijob}')
        return 

    iend = min(istart + files_per_job, nfiles)
    source_scan_ids = source_scan_ids[istart:iend]

    with open(pathname(OUTPUT_LIST_FILE(ijob)), 'w') as fp:
        for id in source_scan_ids:
            fp.write(id + '\n')

# ---- Epilogue -----------------------------
if __name__ == '__main__': 
    main()


