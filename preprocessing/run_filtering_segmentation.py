'''
**A top level script for filtering, resampling and extracting good segments
from the EEG recordings.**
'''
import sys
import glob
import os
import os.path as path
import socket
import commentjson as cjson

from edf_preprocessing import slice_edfs

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)

JSON_CONFIG_FILE = "preproc_conf.json"


def _expand_cfg_path(path_value):
	"""Expand env vars and ~ in path strings from config."""
	return path.expandvars(path.expanduser(path_value))


def _get_host(conf_dict):
	"""Return host key listed in config, or "other" fallback if present."""
	host = socket.getfqdn()

	for key in conf_dict['hosts']:
		if key == 'other':
			continue
		if key in host:
			return key

	if 'other' in conf_dict['hosts']:
		return 'other'

	raise ValueError(f'Host is not listed in the {JSON_CONFIG_FILE}.')


def get_data_folders(conf_dict, out_root_key='out_root'):
	"""Read input/output roots and cluster flag from JSON config.

	Args:
		conf_dict (dict): configuration loaded from preproc_conf.json
		out_root_key (str): key to use for the output root inside each host
			entry (default 'out_root'; use 'hv_out_root' or 'ps_out_root'
			for the HV/PS extract scripts).
	"""
	host = _get_host(conf_dict)
	host_cfg = conf_dict['hosts'][host]

	data_root = _expand_cfg_path(host_cfg['data_root'])
	out_root = _expand_cfg_path(host_cfg[out_root_key])
	cluster_job = bool(host_cfg['cluster_job'])

	# Preserve previous behavior for generic fallback paths.
	if host == 'other':
		if not path.isabs(data_root):
			data_root = path.join(os.getcwd(), data_root)
		if not path.isabs(out_root):
			out_root = path.join(os.getcwd(), out_root)

	return data_root, out_root, cluster_job


def _read_script_config():
	with open(_pathname(JSON_CONFIG_FILE), 'r') as fp:
		return cjson.loads(fp.read())


def _normalize_hospitals(hospital_cfg):
	if isinstance(hospital_cfg, str):
		hospitals = [hospital_cfg]
	elif isinstance(hospital_cfg, list) and all(isinstance(h, str) for h in hospital_cfg):
		hospitals = hospital_cfg
	else:
		raise ValueError('"hospital" must be either a string or a list of strings in preproc_conf.json')

	if not hospitals:
		raise ValueError('"hospital" list in preproc_conf.json should not be empty')

	return hospitals


def _validate_source_scan_ids(source_scan_ids, hospitals):
	if source_scan_ids is None:
		return

	if not isinstance(source_scan_ids, list) or not all(isinstance(s, str) for s in source_scan_ids):
		raise ValueError('"source_scan_ids" must be null or a list of strings in preproc_conf.json')

	if len(hospitals) != 1:
		raise AssertionError('When "source_scan_ids" is provided, exactly one hospital must be specified.')

# ------------------------------------------------------------------
# Main script for filtering, resampling and extracting good segments.
# ------------------------------------------------------------------

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

    data_root, out_root, cluster_job = get_data_folders(conf_dict)

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

        slice_edfs(input_dir, output_dir, conf_dict=conf_dict, source_scan_ids=hospital_scan_ids)

