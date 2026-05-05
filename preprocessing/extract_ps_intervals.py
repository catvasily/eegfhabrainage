'''
**A top level script to perform filtering and extraction of photic stimulation (PS)
intervals.**
'''
import sys
import glob
import os
import os.path as path

from edf_preprocessing import slice_edfs
from run_filtering_segmentation import (
    get_data_folders, _read_script_config,
    _normalize_hospitals, _validate_source_scan_ids
)

# The 'if' is needed to prevent running this code when the file is
# imported into some other source and is not supposed to run
if __name__ == '__main__':
    conf_dict = _read_script_config()

    N_ARRAY_JOBS = int(conf_dict['N_ARRAY_JOBS'])
    if N_ARRAY_JOBS < 1:
        raise ValueError('"N_ARRAY_JOBS" should be >= 1 in preproc_conf.json')

    hospitals = _normalize_hospitals(conf_dict['hospital'])
    source_scan_ids = conf_dict.get('source_scan_ids')
    _validate_source_scan_ids(source_scan_ids, hospitals)

    data_root, out_root, cluster_job = get_data_folders(conf_dict, out_root_key='ps_out_root')

    # When running on the CC cluster, 1st command line argument is a 0-based
    # array job index.
    if len(sys.argv) == 1:
        ijob = 0
    else:
        ijob = int(sys.argv[1])

    for hospital in hospitals:
        input_dir = path.join(data_root, hospital)
        output_dir = path.join(out_root, hospital)

        if not path.exists(output_dir):
            os.makedirs(output_dir)

        if source_scan_ids is None:
            hospital_scan_ids = [path.basename(f)[:-4] for f in glob.glob(input_dir + '/*.edf')]
        else:
            hospital_scan_ids = list(source_scan_ids)

        if cluster_job:
            nfiles = len(hospital_scan_ids)
            files_per_job = nfiles // N_ARRAY_JOBS + 1
            istart = ijob * files_per_job

            if istart > nfiles - 1:
                print(f'All done for {hospital}')
                continue

            iend = min(istart + files_per_job, nfiles)
            hospital_scan_ids = hospital_scan_ids[istart:iend]

        slice_edfs(input_dir, output_dir, conf_dict=conf_dict,
                   source_scan_ids=hospital_scan_ids, extract='PS')

