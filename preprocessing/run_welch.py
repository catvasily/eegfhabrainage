"""
**A top level script to calculate PSDs for sensor and source reconstructed
data using the Welch method.**
"""
import sys
import os
import os.path as path
import glob
import socket
import commentjson as cjson
import numpy as np
import h5py         # Needed to save/load files in .hdf5 format
import matplotlib.pyplot as plt
import mne

from scipy.signal import welch
from do_src_reconstr import read_roi_time_courses
from do_pyprep import Pipeline

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

sys.path.append(pathname('../misc'))
from utils import next_power_of_2, closest_elements

PREPROC_CONFIG_FILE = "preproc_conf.json"
'''Name (without a path) for the JSON file with preprocessing parameters 
which was used to run filtering/segmentation step

'''

WELCH_CONFIG_FILE = 'welch_conf.json'   # This script general configuration settings
INPUT_JSON_FILE = "welch_input.json"    # This script input parameters


def _normalize_hospitals(hospital_cfg):
    if isinstance(hospital_cfg, str):
        hospitals = [hospital_cfg]
    elif isinstance(hospital_cfg, list) and all(isinstance(h, str) for h in hospital_cfg):
        hospitals = hospital_cfg
    else:
        raise ValueError('"hospital" must be either a string or a list of strings in welch_input.json')

    if not hospitals:
        raise ValueError('"hospital" list in welch_input.json should not be empty')

    return hospitals


def _validate_source_scan_ids(source_scan_ids, hospitals):
    if source_scan_ids is None:
        return

    if not isinstance(source_scan_ids, list) or not all(isinstance(s, str) for s in source_scan_ids):
        raise ValueError('"source_scan_ids" must be null or a list of strings in welch_input.json')

    if len(hospitals) != 1:
        raise AssertionError('When "source_scan_ids" is provided, exactly one hospital must be specified.')

def main():
    # Parse input args
    with open(pathname(INPUT_JSON_FILE), 'r') as fp:
        args = cjson.loads(fp.read())

    N_ARRAY_JOBS = args['N_ARRAY_JOBS']     # Number of parallel jobs to run on cluster
    what = args['what']                     # 'sensors' or 'sources'
    hospitals = _normalize_hospitals(args['hospital'])
    source_scan_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf)
    _validate_source_scan_ids(source_scan_ids, hospitals)

    ftype = args['input_file_type']         # fif or edf for sensors, always hdf5 for sources

    if ftype not in ('edf','fif','hdf5'):
        raise ValueError('Unrecognized input file type: ' + ftype)

    if (what == 'sources') and (ftype != 'hdf5'):
        raise ValueError('Source input data files should be in .hdf5 format')

    ico = args['input_file_ico']        # Only used for source-reconstructed data

    view_plots = args['view_plots']         # Flag to show interactive plots
    plot_only = args['plot_only']           # if true, plot already precalculated spectra

    # IDs of channels to plot if view_plots = True, or [] or None
    plot_ids = args['plot_ids']

    # Channels/ROIs to plot, or None, or []
    plot_chnames = args['plot_chnames'][what]

    verbose = args['verbose']               # Can be ‘DEBUG’, ‘INFO', ERROR', 'CRITICAL', or 'WARNING' (default)
    data_root, out_root, cluster_job = get_data_folders(args)
    # ------ end of args parsing ---------------

    mne.set_log_level(verbose=verbose)

    # When running on the CC cluster, 1st command line argument is a 0-based
    # array job index
    if len(sys.argv) == 1:   # No command line args
        ijob = 0
    else:
        ijob = int(sys.argv[1])

    # load the original preprocessing configuration, as we need some data from there
    with open(pathname(PREPROC_CONFIG_FILE), "r") as fp:
        preproc_dict = cjson.loads(fp.read())

    # load this script's configuration parameters
    with open(pathname(WELCH_CONFIG_FILE), "r") as fp:
        conf_dict = cjson.loads(fp.read())

    fs = preproc_dict['target_frequency']   # Sampling frequency
    frange = conf_dict["freq_range"]        # Frequency range
    nf =  conf_dict["freq_points"]          # Number of frequency points (including fmin, fmax)
    freqs, df = np.linspace(frange[0], frange[1], num=nf, endpoint=True, retstep=True)

    # df is the target frequency resolution
    nsamples = int(np.round(fs/df))         # number of samples to collect to achieve df
    nfft = next_power_of_2(nsamples)        # FFT length for each segment in Welch
    idx = None

    for hospital in hospitals:
        input_dir = data_root + "/" + hospital
        output_dir = out_root + "/" + hospital

        if not path.exists(output_dir):
            os.makedirs(output_dir)

        if source_scan_ids is None:
            if what == 'sensors':
                if ftype == 'fif':
                    # To get bare ID need to chop off "_raw.fif" at the end
                    hospital_scan_ids = [path.basename(f)[:-8] for f in glob.glob(input_dir + '/*.fif')]
                else:       # ftype == 'edf':
                    # To get bare ID need to chop off ".edf" at the end
                    hospital_scan_ids = [path.basename(f)[:-4] for f in glob.glob(input_dir + '/*.edf')]
            else:
                # To get bare ID one needs to get folders with names that are 5 HEX numbers
                # separated by 4 dashes. Poor man's solution for it is just '*-*-*-*-*'
                hospital_scan_ids = [path.basename(f) for f in glob.glob(input_dir + '/*-*-*-*-*')]
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

        if what == 'sensors':
            if ftype == 'fif':
                scan_files = [scan_id + '_raw.fif' for scan_id in hospital_scan_ids]
            else:       # ftype == 'edf':
                scan_files = [scan_id + '.edf' for scan_id in hospital_scan_ids]
        else:
            scan_files = [scan_id + '/' + scan_id + f'-ico-{ico}-ltc.hdf5' \
                    for scan_id in hospital_scan_ids]

        outname = lambda scan_id:   \
                output_dir + '/' + scan_id + \
                ('_psd.hdf5' if what == 'sensors' else  '_src_psd.hdf5')

        if not plot_only:
            # main loop
            print('\nProcessing record IDs:')
            for scan_id, f in zip(hospital_scan_ids, scan_files):
                print(f'{scan_id}')
                data, ch_names = load_data(input_dir + '/' + f)

                f, Pxx = welch(data, fs=fs, nperseg=nfft, **conf_dict['welch'])

                if idx is None:
                    idx = closest_elements(f, freqs)    # indecies of f's that are close to freqs

                Pxx = Pxx[:, idx]
                write_welch_psd(outname(scan_id), ch_names, freqs, Pxx)

            print(f'Total {len(hospital_scan_ids)} records processed for {hospital}\n')

        if view_plots and (plot_ids is not None):
            for pid in plot_ids:
                plt.figure()
                plot_psd(outname(pid), plot_chnames, title = pid)

            plt.show()

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

    raise ValueError(f'Host is not listed in the {INPUT_JSON_FILE}.')

def get_data_folders(args):
    '''Setup input and output data folders depending on the host machine.

    Args:
        args (dict): dictionary with input arguments read from the 
            `INPUT_JSON_FILE`

    Returns:
        data_root (str): path to the root input folder
        out_root (str): path to the root output folder
        cluster_job ( bool): a flag indicating whether the host is on CC cluster
    '''
    what = args['what']

    valid_whats = {'sensors', 'sources'}
    if what not in valid_whats:
        raise ValueError(f'Invalid arguement \'{what}\' passed; should be one of {valid_whats}')

    # path.expanduser("~") results in /home/<username>
    # user_home = path.expanduser("~")
    # user = path.basename(user_home) # Yields just <username>

    # Choose appropriate host name from those listed in the json:
    host_found = False
    host = _get_host(args)

    for key in args['hosts']:
        if key in host:
            host = key
            host_found = True
            break

    if not host_found:
        host = 'other'

    # Get the host data
    cluster_job = args['hosts'][host].get('cluster_job', False)
    data_root = args['hosts'][host][what]['data_root']
    out_root = args['hosts'][host][what]['out_root']

    # Additional adjustments
    if not cluster_job:
        mne.viz.set_browser_backend('matplotlib')

    if host == 'other':
        work_dir = os.getcwd()
        data_root = work_dir + '/' + data_root
        out_root = work_dir + '/' + out_root

    return data_root, out_root, cluster_job

def load_data(f):
    '''Read signal data from .fif or .hdf5 file

    Args:
        f (str): full pathnname of the data file

    Returns:
        data (nparray): `nchan x ntimes` signal data array
        chnames (list of str): channel names
    '''
    def load_fif(f):
        raw = mne.io.read_raw_fif(f, preload=True)
        raw.pick('data', exclude = 'bads')
        return raw.get_data(), raw.ch_names

    def load_hdf5(f):
        # the returned tuple is: (label_tcs, label_names, vertno, rr, W, pz)
        return read_roi_time_courses(f)[:2]

    def load_edf(f):
        # Reuse Pipeline class init() code as it properly updates EDF channel types
        raw = Pipeline(f).getRaw()
        mne.datasets.eegbci.standardize(raw)    # Change chnames to standard; i.e. PZ -> Pz
        return raw.get_data(), raw.ch_names

    # The main function body start
    ext = path.splitext(f)[1]

    if ext == '.fif':
        return load_fif(f)
    elif ext == '.edf':
        return load_edf(f)
    elif ext == '.hdf5':
        return load_hdf5(f)
    else:
        raise ValueError('Unrecognized input file type')

def write_welch_psd(fname, ch_names, freqs, psd):
    """Save PSDs of sensor or reconstructed ROI signals in .hdf5
    file.

    The output file will contain three datasets: 'ch_names' with sensor channel or
    ROI names, 'freqs' with frequency values in Hz, and 'psd' with PSD values in 
    units of U^2/Hz, where U is Volts for sensor channels PSD. For ROI PSDs the unit U
    depends on the setting of `src_units` parameter in `src_reconstr_conf.json` file
    at the time of running the source reconstruction step. It is A*m when `src_units`
    is set to 'source', or amplitude pseudo-Z when it is set to 'pz'.

    Args:
        fname (str): full pathname of the output .hdf5 file. 
        ch_names (list of str): names of channels or ROIs
        freqs (ndarray): shape (nf,) frequency values, Hz
        psd (ndarray): shape (nchans, nf) PSD values in U^2/Hz

    Returns:
        None
    """
    if path.exists(fname):
        raise ValueError(f'Output file {fname} already exists. Please delete or move it.')

    with h5py.File(fname, 'w') as f:
        f.create_dataset('ch_names', data=ch_names)
        f.create_dataset('freqs', data=freqs)
        f.create_dataset('psd', data=psd)

def read_welch_psd(fname):
    """Read PSDs of sensor or reconstructed ROI signals in .hdf5
    file.

    The input file is expected to contain three datasets: 'ch_names' with sensor channel or
    ROI names, 'freqs' with frequency values in Hz, and 'psd' with PSD values in 
    units of U^2/Hz, where U is Volts for sensor channels PSD. For ROI PSDs the unit U
    depends on the setting of `src_units` parameter in `src_reconstr_conf.json` file
    at the time of running the source reconstruction step. It is A*m when `src_units`
    is set to 'source', or amplitude pseudo-Z when it is set to 'pz'.

    Args:
        fname (str): full pathname of the input .hdf5 file. 

    Returns:
        ch_names (list of str): names of channels or ROIs
        freqs (ndarray): shape (nf,) frequency values, Hz
        psd (ndarray): shape (nchans, nf) PSD values in U^2/Hz
    """
    with h5py.File(fname, 'r') as f:
        ch_names = f['ch_names'].asstr()[:]
        freqs = f['freqs'][:]
        psd = f['psd'][:,:]

    return ch_names, freqs, psd

def plot_psd(fname, ch_names, title = None, xlim = None, ylim = None):
    '''Plot saved sensor or ROI signals PSDs

    Args:
        fname (str): full pathname to the .hdf5 file created by
            `write_welch_psd()`
        ch_names (list of str): list of channel or ROI names to plot
        title (str or None): plot title
        xlim ([xmin, xmax] or None): X-axis limits
        ylim ([ymin, ymax] or None): Y-axis limits 

    Returns:
        Nothing
    '''

    chs, freqs, psd = read_welch_psd(fname)
    idx = [list(chs).index(ch) for ch in ch_names]
    plt.semilogy(freqs, psd[idx,:].T)
    plt.legend(ch_names)
    plt.xlabel('Hz')
    plt.ylabel('Pwr/Hz')

    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is None:
        ymin, ymax = plt.ylim()
        plt.ylim(ymax*1e-3, ymax)
    else:
        plt.ylim(ylim)

    # plt.show()

if __name__ == '__main__': 
    main()


