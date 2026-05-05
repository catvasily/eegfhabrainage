"""
**A top level script to view various mutli-dimensional data embedded in
2D or 3D space.**

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
INPUT_JSON_FILE = '<my-file>.json
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

`'input'`:
  As is: set all the input and configuration parameters here. This step
  should always be run first  

`'rois_spectral_power'`:
  For each EEG record, construct a vector of values of total spectral power
  in a specified frequency band for all   atlas ROIs. Visualize spatial
  distribution of these vectors in nROIs-dimensional space

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

from run_welch import read_welch_psd

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

sys.path.append(pathname('../misc'))
from view_hd_embedding import view_hd_embedding

INPUT_JSON_FILE = "view_hd_input.json"      # This script's input parameters

def inputs(ss):
    """
    Read and process common input parameters from JSON file.
    """
    args = ss.args
    source_scan_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf, .fif, etc)
    view_plots = args['view_plots']         # Flag to show interactive plots
    plot_only = args['plot_only']           # if true, plot already precalculated spectra
    what = args['what']                     # 'sensors' or 'sources'
    lst_hosp = args['hospitals']            # A list of hospitals to process

    data_root, out_root, cluster_job = get_data_folders(args)
    # ------ end of args parsing ---------------

    # Get lists of scan IDs for each hospital as a dictionary
    # {hospital: list_of_IDs}
    if source_scan_ids is not None:
        dict_ids = {hospital: source_scan_ids[i] for i,hospital in enumerate(lst_hosp)}
    else:
        dict_ids = dict()
        for hospital in lst_hosp:
            input_dir = data_root + "/" + hospital
    
            if what == 'sensors':
                # To get bare ID need to chop off "_psd.hdf5" at the end
                lst_ids = [path.basename(f)[:-9] for f in glob.glob(input_dir + '/*.hdf5')]
            else:
                # To get bare ID need to chop off "_src_psd.hdf5" at the end
                lst_ids = [path.basename(f)[:-13] for f in glob.glob(input_dir + '/*.hdf5')]

            dict_ids[hospital] = lst_ids

    if not path.exists(out_root):
        os.makedirs(out_root)

    if cluster_job:
        view_plots = False

    # Save useful info as ss attributes
    ss.dict_ids = dict_ids
    ss.data_root = data_root
    ss.out_root = out_root
    ss.view_plots = view_plots

    # Save total number of IDs to process
    ss.n_ids = sum([len(dict_ids[h]) for h in dict_ids])

def rois_spectral_power(ss):
    """
    For each EEG record, construct a vector of values of total
    spectral power in a specified frequency band for all
    atlas ROIs. Visualize spatial distribution of these vectors
    in nROIs-dimensional space by embedding in low dim space.
    """
    def initialize_hd_data(fname):
        """
        Determine required dimensions, and then construct the
        hd_data array
        """
        nROIs = len(read_welch_psd(fname)[0])
        return np.zeros((ss.n_ids, nROIs))

    if ss.args['what'] == 'sensors':
        # For what == 'sensors' one needs to deal with channel sets
        # being different for each record
        raise NotImplementedError('Setting what = "sensors" is not yet implemented')

    hd_data = None
    i = 0   # Scan ID counter

    for ih, hospital in enumerate(ss.args['hospitals']):
        input_dir = ss.data_root + "/" + hospital

        for sid in ss.dict_ids[hospital]:
            fname = input_dir + f'/{sid}_src_psd.hdf5'

            if hd_data is None:
                hd_data = initialize_hd_data(fname)
                taxonomy = np.zeros(ss.n_ids, dtype = int)

            hd_data[i] = band_power(ss, fname)
            taxonomy[i] = ih
            i += 1

    # Embed and display hd_data
    labels = {i:h for i,h in enumerate(ss.args['hospitals'])}
    args = ss.args['rois_spectral_power']['view_hd_embedding'].copy()
    band = ss.args['rois_spectral_power']['band']
    args['title'] = args['method'] + f': {band[0]} - {band[1]}' + args['title']
    args['save_file'] = ss.out_root + '/' + args['method'] + \
        f'_{band[0]}-{band[1]}' + args['save_file']

    view_hd_embedding(hd_data, taxonomy = taxonomy, seed = ss.args['rand_seed'],
        labels = labels, show_plot = ss.view_plots, **args)

def band_power(ss, fname):
    """
    Calculate power in a specified band for each ROI.

    Args:
        ss (obj): reference to this app object
        fname(str): full path to .hdf5 file with source-space spectral data

    Returns:
        spect(ndarray): shape (nROIs,) vector of band powers for the ROIs

    """
    band = ss.args["rois_spectral_power"]["band"]
    _, f, psd = read_welch_psd(fname)   # psd is nchans x nf
    idx=np.logical_and(f>=band[0], f<=band[1])
    spect = np.mean(psd[:,idx], axis = 1)    # spect is a 1D array of n_ROI points
    
    return spect

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
        'inputs': inputs,
        'rois_spectral_power': rois_spectral_power,
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

    with open(pathname(INPUT_JSON_FILE), 'r') as fp:
        this_app.args = cjson.loads(fp.read())

    for name in this_app.args['to_run']:     
        this_app(name)

# -------------- end of Epilogue --------------------------


