"""
**A top level script for running spectra parameterization into
periodic and aperiodic components.**

Based on code in  

https://github.com/fooof-tools/fooof  
https://pypi.org/project/specparam/     # The latest package   

and original Nature Neuroscience paper referenced therein.

To add a step to be run to this script:

1. Encapsulate necessary code in a dedicated function  

```
def my_step(ss):  
...
# NOTE: Step return value (if any) is not used
```

2. Add corresponding entry to the cases dictionary in
the EPILOGE section at the bottom in the form  

```
'my_step': my_step
```

Steps may reside in separate Python files. In that case
corresponding import statements should be added here.
The variables that are intended to be shared between
steps should be defined as attributes of the ss object,
as follows:

```
ss.common_var = common_var_value
```

All input parameters for the script are expected to reside
in a single JSON file specifed in a global constant 
`INPUT_JSON_FILE` as follows:

```
INPUT_JSON_FILE = '<my-file>.json'
```

Contents of the INPUT_JSON_FILE is available to each step
as a dictionary `ss.args`, where `ss` is a reference to this
application object passed to the step as an argument.

The sequence of steps to be executed should be listed in `to_run`
key in the JSON input file, for example
```
to_run = ("init","step1","step6")
```

--------------------------------------------------

Available steps:

`'input'`: As is: set all the input and configuration
  parameters here. This step should always be run first  
`'do_fit'`: Run model fitting for all requested scan IDs and
  save the results to .hdf5 files.
`cumulative_report`: Create a cumulative report for all processed scans
"""

import sys
import os
import os.path as path
import io
import glob
import socket
import matplotlib.pyplot as plt
import commentjson as cjson
import numpy as np
import h5py
import specparam
from specparam import SpectralModel, SpectralGroupModel
from run_welch import read_welch_psd

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)  # full pathname to source file

# These are only needed for benchmarking
sys.path.append(path.dirname(path.dirname(__file__))+ "/misc")
from utils import timeit

#-------------------------------------
INPUT_JSON_FILE = "spectparam_input.json"      # This script's input parameters
#-------------------------------------

def input(ss):
    '''
    Read and process common input parameters from JSON file.
    '''
    args = ss.args
    hospital = args['hospital']            # A list of hospitals to process
    lst_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf, .fif, etc)
    what = args['what']                     # 'sensors' or 'sources'

    data_root, out_root, cluster_job = get_data_folders(args)
    # ------ end of args parsing ---------------

    ss.input_dir = data_root + '/' + hospital

    if lst_ids is None:
        if what == 'sensors':
            # To get bare ID need to chop off "_psd.hdf5" at the end
            lst_ids = [path.basename(f)[:-9] for f in glob.glob(ss.input_dir + '/*.hdf5')]
        else:
            # To get bare ID need to chop off "_src_psd.hdf5" at the end
            lst_ids = [path.basename(f)[:-13] for f in glob.glob(ss.input_dir + '/*.hdf5')]

    ss.out_dir = out_root + '/' + hospital
    os.makedirs(ss.out_dir, mode = 0o775, exist_ok = True)

    # When running on the CC cluster, 1st command line argument is a 0-based
    # array job index. lst_ids will be different for each job
    if cluster_job:
        nfiles = len(lst_ids)
        files_per_job = nfiles // args['N_ARRAY_JOBS'] + 1
        ijob = int(sys.argv[1])         # Job number is passed as 1st cmd line arg
        istart = ijob * files_per_job

        if istart > nfiles - 1:
            print('All done')
            sys.exit()

        iend = min(istart + files_per_job, nfiles)
        lst_ids = lst_ids[istart:iend]

    # Pass common data to other steps
    ss.lst_ids = lst_ids

def _input_fname(ss, sid):
    '''
    Generate full file name for input power spectrum for given scan ID
    '''
    in_suffix = '_psd.hdf5' if ss.args['what'] == 'sensors' else '_src_psd.hdf5'
    return ss.input_dir + '/' + sid + in_suffix

def _output_fname(ss, sid):
    '''
    Generate full output file name for model fit results for given scan ID
    '''
    out_suffix = '_psd_model.hdf5' if ss.args['what'] == 'sensors' else '_src_psd_model.hdf5'
    return ss.out_dir + '/' + sid + out_suffix

@timeit
def do_fit(ss):
    '''
    Perform model fits for specified scans.
    '''

    print('Processing scan IDs:')
    for sid in ss.lst_ids:
        print(sid)
        fname = _input_fname(ss, sid);
        ch_names, freqs, psd = read_welch_psd(fname)   # psd = nchan x nf 

        # Check for possible zeros in PSDs to avoid log10(0) exception
        psd[psd < 0.5e-100] = 1e-100

        # Initialize a SpectralGroupModel object, specifying some parameters
        fg = SpectralGroupModel(**ss.args['model'])

        # Fit models across the matrix of power spectra
        fg.fit(freqs, psd, n_jobs = -1)

        # Save results to HDF file
        outname = _output_fname(ss, sid)
        write_model(outname, ch_names, fg);

    print(f'\nSuccessfully processed {len(ss.lst_ids)} records.')

def cumulative_report(ss):
    '''
    Create a cumulative report for all processed scans, either for all channels at once
    or for specified channels only.
    '''
    def get_channel_nums(lst):
        '''
        Convert channel names list to channel indecies.
        NOTE: Throws an exception if invalid channel name is encountered
        '''
        if lst is None:
            return list(range(len(ch_names)))

        return [list(ch_names).index(ch) for ch in lst]
    # -----------------------------

    lst_idx = None      # Indecies of channels to be included into report
    lst_ids = _pick_SIDs_for_report(ss)

    for sid in lst_ids:
        fname = _output_fname(ss, sid)

        if lst_idx is None:
            ch_names, freqs, model_settings, lst_results = read_model(fname)
            lst_idx = get_channel_nums(ss.args['report']['report_channels'])
            fg = SpectralGroupModel(*model_settings)

            freq_range = [freqs[0], freqs[-1]]
            freq_res = freqs[1] - freqs[0]
            meta_data = specparam.data.SpectrumMetaData(freq_range, freq_res)
            fg.add_meta_data(meta_data)

        else:
            lst_results = read_model(fname)[3]  # Get a list of nchan FitResults objects

        for ich in lst_idx:
            fit = SpectralModel(*model_settings)
            fit.add_meta_data(meta_data)
            fit.add_results(lst_results[ich])
            fg = specparam.objs.combine_model_objs([fg, fit])

    file_name = ss.out_dir + '/' + ss.args['report']['file_png']
    fg.save_report(file_name)
    print(f'\nFinished creating cumulative report. Report saved to file {file_name}')

def _pick_SIDs_for_report(ss):
    '''
    Randomly select `max_num_scan_ids` from all available scan IDs. If the
    total is less or equal than `max_num_scan_ids`, simply return all scan
    IDs
    '''
    m = ss.args['report']['max_num_scan_ids']

    if len(ss.lst_ids) <= m:
        return ss.lst_ids

    if not hasattr(ss, 'rng'):
        ss.rng = np.random.default_rng(ss.args['report']['seed'])

    return ss.rng.choice(ss.lst_ids, size=m, replace=False)

def write_model(fname, ch_names, fg):
    '''
    Save model fit results to HDF file.
    The data which is saved includes: channel names, a list of frequencies,
    model settings (`peak_width_limits`, `max_n_peaks`, `min_peak_height`, 
    `peak_threshold`, `aperiodic_mode`) and fit results for each channel
    as returned in corresponding FitResults object.

    Args:
        fname (str): pathname to the output .hdf5 file to be created
        ch_names (lst of str): list of channel names
        fg(SpectralGroupModel): a model group object with fit results

    Returns:
        Nothing
    '''
    with h5py.File(fname, 'w') as f:
        # Channels and frequencies
        f.create_dataset('ch_names', data=ch_names)
        f.create_dataset('freqs', data=fg.freqs)

        # Save settings
        peak_width_limits, max_n_peaks, min_peak_height, \
            peak_threshold, aperiodic_mode = fg.get_settings()

        f.attrs['min_peak_width'] =  peak_width_limits[0]
        f.attrs['max_peak_width'] =  peak_width_limits[1]
        f.attrs['max_n_peaks'] = max_n_peaks
        f.attrs['min_peak_height'] = min_peak_height
        f.attrs['peak_threshold'] = peak_threshold
        f.attrs['aperiodic_mode'] = aperiodic_mode

        fit_results = _fitres2array(fg)
        f.create_dataset('fit_results', data=fit_results)

def read_model(fname):
    '''
    Read the model fit results from an HDF file created using `write_model()`
    The data which is returned includes: channel names, a list of frequencies
    model settings (`peak_width_limits`, `max_n_peaks`, `min_peak_height`, 
    `peak_threshold`, `aperiodic_mode`), and fit results for each channel as
    a list of corresponding FitResults objects.

    Args:
        fname (str): pathname to the input .hdf5 file
        fg(SpectralGroupModel): a model group object with fit results

    Returns:
        ch_names (lst of str): list of channel names
        freqs(ndarray): list of frequencies
        model_settings (ModelSettings): settings object, which is a named
            tuple of `peak_width_limits, max_n_peaks, min_peak_height,
            peak_threshold, aperiodic_mode`
        lst_results (list of FitResults): fit results for each channel 
    '''
    with h5py.File(fname, 'r') as f:
        ch_names = f['ch_names'].asstr()[:]
        freqs = f['freqs'][:]

        peak_width_limits = (f.attrs['min_peak_width'], f.attrs['max_peak_width'])
        max_n_peaks = f.attrs['max_n_peaks']
        min_peak_height = f.attrs['min_peak_height']
        peak_threshold = f.attrs['peak_threshold']
        aperiodic_mode = f.attrs['aperiodic_mode']
        model_settings = specparam.data.ModelSettings(peak_width_limits, max_n_peaks,
                min_peak_height, peak_threshold, aperiodic_mode)

        data = f['fit_results'][:]
        lst_results = _array2fitres(data, len(ch_names))

    return ch_names, freqs, model_settings, lst_results

def _fitres2array(fg):
    '''
    Utility function that serializes fit results from the SpectralGroupModel
    object to a 1D numpy array. The array is layed out as follows:

    `results for chan 0`  
        ...
    `results for chan <nchan-1>  

    Results for each channel are layed out as follows:  
    `float(n_found_peaks), aperiodic_params, peak_params,
    r_squared, error, gaussian_params` 

    '''
    fits = fg.get_results()
    lst = []

    for f in fits:
        aperiodic_params, peak_params, r_squared, error, gaussian_params = f
        npeaks = len(peak_params)
        lst.extend([float(npeaks), *aperiodic_params, *(peak_params.flatten()),
            r_squared, error, *(gaussian_params.flatten())])

    return np.array(lst)

def _array2fitres(data, nchans):
    '''
    Utility function to restore fit results serialized with `_fitres2array()`
    into the 1D numpy array. Returns a list of `FitResults objects` - one 
    object per channel.
    '''
    i = 0
    lst = []

    for ichan in range(nchans):
        npeaks = int(data[i]); i += 1
        aperiodic_params = data[i:i+2].copy(); i += 2

        lenpeaks = npeaks*3     # Total length of all peak parameters
        peak_params = np.reshape(data[i:i + lenpeaks],(npeaks, 3), order = 'C')
        i += lenpeaks

        r_squared = data[i]; i += 1
        error = data[i]; i += 1

        gaussian_params = np.reshape(data[i:i + lenpeaks],(npeaks, 3), order = 'C')
        i += lenpeaks

        lst.append(specparam.data.FitResults(aperiodic_params, peak_params,
            r_squared, error, gaussian_params))

    return lst

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
        raise ValueError(f'Invalid argument \`{what}\` passed; should be one of {valid_whats}')

    # path.expanduser("~") results in /home/<username>
    # user_home = path.expanduser("~")
    # user = path.basename(user_home) # Yields just <username>

    # Choose appropriate host name from those listed in the json:
    host_found = False
    host = socket.gethostname()

    for key in args['hosts']:
        if key in host:
            host = key
            host_found = True
            break

    if not host_found:
        host = 'other'

    # Get the host data
    cluster_job = args['hosts'][host]['cluster_job']
    data_root = args['hosts'][host][what]['data_root']
    out_root = args['hosts'][host][what]['out_root']

    if host == 'other':
        work_dir = os.getcwd()
        data_root = work_dir + '/' + data_root
        out_root = work_dir + '/' + out_root

    return data_root, out_root, cluster_job

# --------------------------------------------------------
#                    EPILOGUE                             
# --------------------------------------------------------
class _app:
    # ------
    # Cases: add your steps here in the form "my_step":my_step,
    # ------
    cases = {
        # Steps to run go here:
        'input': input,
        'do_fit': do_fit,
        'cumulative_report': cumulative_report,
    }

    def __call__(self, name, *args, **kwargs):
        not_found = True

        for f in self.cases:
            if f == name:
                self.cases[f](self, *args, **kwargs)
                not_found = False
                break

        if not_found:
            raise ValueError(f'Requested method "{name}" not found')

if __name__ == '__main__': 
    for c in _app.cases:
        setattr(_app, c, _app.cases[c])

    this_app = _app()

    with open(_pathname(INPUT_JSON_FILE), 'r') as fp:
        this_app.args = cjson.loads(fp.read())

    for name in this_app.args['to_run']:     
        this_app(name)

# -------------- end of Epilogue --------------------------


