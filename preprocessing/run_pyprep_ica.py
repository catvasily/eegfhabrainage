'''
**A top level script to perform the PyPREP and ICA artifact removal step.**
'''
import sys
import glob
import os
import os.path as path
import socket
import mne
import commentjson as cjson

from  do_pyprep import Pipeline

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)

JSON_CONFIG_FILE = "pyprep_ica_conf.json"


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


def get_data_folders(conf_dict):
    """Read input/output roots and cluster flag from JSON config."""
    host = _get_host(conf_dict)
    host_cfg = conf_dict['hosts'][host]

    data_root = _expand_cfg_path(host_cfg['data_root'])
    out_root = _expand_cfg_path(host_cfg['out_root'])
    cluster_job = bool(host_cfg['cluster_job'])

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
        raise ValueError('"hospital" must be either a string or a list of strings in pyprep_ica_conf.json')

    if not hospitals:
        raise ValueError('"hospital" list in pyprep_ica_conf.json should not be empty')

    return hospitals


def _validate_source_scan_ids(source_scan_ids, hospitals):
    if source_scan_ids is None:
        return

    if not isinstance(source_scan_ids, list) or not all(isinstance(s, str) for s in source_scan_ids):
        raise ValueError('"source_scan_ids" must be null or a list of strings in pyprep_ica_conf.json')

    if len(hospitals) != 1:
        raise AssertionError('When "source_scan_ids" is provided, exactly one hospital must be specified.')

if __name__ == '__main__': 
    conf_dict = _read_script_config()

    N_ARRAY_JOBS = int(conf_dict['N_ARRAY_JOBS'])
    if N_ARRAY_JOBS < 1:
        raise ValueError('"N_ARRAY_JOBS" should be >= 1 in pyprep_ica_conf.json')

    hospitals = _normalize_hospitals(conf_dict['hospital'])
    source_scan_ids = conf_dict.get('source_scan_ids')
    _validate_source_scan_ids(source_scan_ids, hospitals)

    view_plots = bool(conf_dict.get('view_plots', False))
    verbose = conf_dict.get('verbose', 'WARNING')

    data_root, out_root, cluster_job = get_data_folders(conf_dict)

    mne.viz.set_browser_backend('matplotlib')
    mne.set_log_level(verbose=verbose)

    # When running on the CC cluster, 1st command line argument is a 0-based
    # array job index
    if len(sys.argv) == 1:   # No command line args
        ijob = 0
    else:
        ijob = int(sys.argv[1])

    for hospital in hospitals:
        input_dir = path.join(data_root, hospital)
        output_dir = path.join(out_root, hospital)
        png_path = output_dir + '/'

        if not path.exists(output_dir):
            os.makedirs(output_dir)

        if source_scan_ids is None:
            hospital_scan_ids = [path.basename(f)[:-4] for f in glob.glob(input_dir + '/*.edf')]
        else:
            hospital_scan_ids = list(source_scan_ids)

        if cluster_job:
            view_plots = False    # Disable interactive plots, just in case
            nfiles = len(hospital_scan_ids)
            files_per_job = nfiles // N_ARRAY_JOBS + 1
            istart = ijob * files_per_job

            if istart > nfiles - 1:
                print(f'All done for {hospital}')
                continue

            iend = min(istart + files_per_job, nfiles)
            hospital_scan_ids = hospital_scan_ids[istart:iend]

        scan_files = [scan_id + '.edf' for scan_id in hospital_scan_ids]

        success = True
        for i, f in enumerate(scan_files):
            filepath = input_dir + '/' + f
            scan_id = hospital_scan_ids[i]
            png_prefix = png_path + scan_id

            ts_org_png = png_prefix + "_ts_org.png"
            psd_org_png = png_prefix + "_psd_org.png"
            ts_postprep_png = png_prefix + "_ts_postprep.png"
            psd_postprep_png = png_prefix + "_psd_postprep.png"
            psd_postica_png = png_prefix + "_psd_postica.png"

            try:
                # Initiate the preprocessing object
                p = Pipeline(filepath, conf_dict=conf_dict, view_plots = view_plots,
                             ts_plot_file = ts_org_png, psd_plot_file = psd_org_png)

                # Apply PREP and ICA
                p.applyPipeline(applyICA = True, view_plots = view_plots,
                    ts_postprep_png = ts_postprep_png, psd_postprep_png = psd_postprep_png,
                    psd_postica_png = psd_postica_png)

                # Get the resulting mne.Raw object
                raw = p.getRaw()

                # Keep the bad channels just in case, and save data in .fif file
                output_path = output_dir + '/' + f[:-4] + '_raw.fif'
                raw.save(fname = output_path, proj = False, fmt = 'single', overwrite = True)
                print('\n***** Processing of {} completed\n'.format(f), flush = True)
            except Exception as e:
                success = False
                print('\n***** Record {} !!! FAILED !!!'.format(f))
                print(e, flush = True)
                print('\n')

        print("\n{} files processed for {} {}.".format(len(scan_files), hospital,
              'successfully' if success else 'with errors'))

