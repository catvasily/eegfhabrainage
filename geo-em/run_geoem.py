"""
**A top level script for running joint EEG and geo-electromagnetic
data analyses.**

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
    `'input'`: As is - set all the input and configuration
        parameters here. This step should always be run first
    `'dates_distr'`: plot records' dates distribution

"""

import sys
import os
import os.path as path
import io
import glob
import socket
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import commentjson as cjson
import numpy as np
import h5py
import specparam
from specparam import SpectralModel, SpectralGroupModel

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)  # full pathname to source file

# These are needed to access specparam functions
sys.path.append(path.dirname(path.dirname(__file__))+ '/preprocessing')
sys.path.append(path.dirname(path.dirname(__file__))+ '/misc')

from run_spectparam import read_model
from data_to_pca import data_to_pca

#-------------------------------------
INPUT_JSON_FILE = "geoem_input.json"      # This script's input parameters
#-------------------------------------

def input(ss):
    """
    Read and process common input parameters from JSON file.

    Args:
        ss(obj): reference to this app object

    """
    args = ss.args
    hospital = args['hospital']            # A list of hospitals to process
    lst_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf, .fif, etc)
    what = args['what']                     # 'sensors' or 'sources'

    data_root, out_root, meta_csv, geoem_dat, cluster_job = get_data_folders(args)
    # ------ end of args parsing ---------------

    ss.input_dir = data_root + '/' + hospital

    if lst_ids is None:
        if what == 'sensors':
            # To get bare ID need to chop off "_psd_model.hdf5" at the end
            lst_ids = [path.basename(f)[:-15] for f in glob.glob(ss.input_dir + '/*.hdf5')]
        else:
            # To get bare ID need to chop off "_src_psd_model.hdf5" at the end
            lst_ids = [path.basename(f)[:-19] for f in glob.glob(ss.input_dir + '/*.hdf5')]

    ss.out_dir = out_root + '/' + hospital
    # QQQ os.makedirs(ss.out_dir, mode = 0o775, exist_ok = True)

    ss.meta_csv = meta_csv
    ss.geoem_dat = geoem_dat
    ss.cluster_job = cluster_job

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

    # Read metadata: fill ss.meta_df DataFrame
    read_metadata(ss)

def read_metadata(ss):
    """
    Read metadata from the cumulative .csv file into a dataframe,
    then only keep records in this dataframe which correspond to
    seclected hospital and belong to ss.lst_ids list.

    Additionally, the time value for each record is rounded
    to the nearest hour.

    Args:
        ss(obj): reference to this app object

    Returns a filled dataframe as `ss.meta_df` 

    """
    df = pd.read_csv(ss.meta_csv)
    parms = ss.args['meta_data']

    # Keep only out-patient records, if requested
    if parms['out_patients_only']:
        df = df[df['AdmittingStatus'] != parms['in_patient_status']]
        ptype = 'out-patients'
    else:
        ptype = 'both in- and out-patients'

    # Only keep meta data for the records at hand
    lst_hospitals = [ss.args['hospital']]
    df = df[df[parms['hospital']].isin(lst_hospitals)]
    df = df[df[parms['scan_id']].isin(ss.lst_ids)]

    # Convert scan dates to datetime format
    date_col = parms['scan_date_col']
    time_col = parms['scan_time_col']
    df[date_col] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), \
            format = '%Y-%m-%d %H:%M:%S')

    # Keep the true date time in ActualDatetime column
    df = df.rename(columns={date_col: 'ActualDatetime'})

    # Have rounded to an hour datetime as the Datetime column
    df['Datetime'] = df['ActualDatetime'].dt.round('H')

    df.drop(columns = [time_col], inplace = True)

    # Sort metadate by the date
    df = df.sort_values(by='Datetime')
    ss.meta_df = df
    print(f'Read EEG meta data for {len(df)} {ptype} records')

def read_geoem_data(ss):
    """
    Load subset of geomagnetic data from a text (.dat) file which is in fact in a
    .CSV format into a dataframe.

    Args:
        ss(obj): reference to this app object

    Returns:
        ss.geoem_df(DataFrame): a data frame with a set of columns specified in the input
            JSON file
    """
    parms = ss.args['geoem_data']
    cols_to_keep = np.array(parms['get_cols_idx']) - 1  # Convert to 0-based indices
    sep = parms['sep']

    if sep == ' ':  # If white space is given as separator - merge all white space
        sep = r'\s+'
    
    df = pd.read_csv(ss.geoem_dat, sep = sep, usecols = cols_to_keep, header = None)

    # Convert all columns past the first three (Year, Day, Hour) to floating point type,
    # because we'll be using NaNs (i.e. float64) for missing values

    FLOAT_COL_IDX_START = 3     # 0-based starting index of cols converted to float
    df = pd.concat([df.iloc[:,:FLOAT_COL_IDX_START], \
            df.iloc[:,FLOAT_COL_IDX_START:].astype('float64')], axis=1)

    # Insert NaNs for missing data
    lstNaNs = parms['get_cols_nans']   # List of missing value markers
    for icol in range(len(lstNaNs)):
        df.iloc[:,icol] = df.iloc[:,icol].replace(lstNaNs[icol], np.nan)

    # Assign column names to the DataFrame; note that at the moment we have
    # Year, Day and Hour columns instead of a single Datetime column
    df.columns = ['Year', 'Day', 'Hour'] + parms['df_col_names'][1:]

    # Add a Datetime column in datetime format
    df['Datetime'] = pd.to_datetime(df['Year'].astype(str) + ' ' + df['Day'].astype(str) + \
            ' ' + df['Hour'].astype(str), format='%Y %j %H')

    # Drop Year, Day, Hour columns
    df.drop(columns = ['Year', 'Day', 'Hour'], inplace = True)
    ss.geoem_df = df
    print(f'Read {len(df)} records of geo_em data')

def read_specparam_data(ss):
    """
    Load 'specparam' (power spectrum model fit) data for specified list of 
    scan IDs and create a dataframe with selected subset of fitted parameters.
    This set is specified in a dedicated key of the input JSON file.

    Note that model fit is performed separately for each channel of the EEG
    record. Assuming that a record has `nchan` channels, and the requested
    subset contains  parameters `a,b,...`, the dataframe will have columns

    `scan_id, a0, b0, ..., a1, b1, ..., a<nchan-1>, b<nchan-1>, ...`

    Thus the total number of columns is `ncols = 1 + nparms * nchan`.

    Args:
        ss(obj): reference to this app object

    Returns:
        ss.specparam_df(DataFrame): a data frame with fitted parameter values
            for each scan ID

    """
    conf = ss.args['specparam']     # Related settings from JSON file

    read_all = True     # Flag to save all data read from .hdf5
                        # Only need to do this once or not need at all
    make_headers = True # Flag to construct column headers
    headers = ['scan_id']
    alpha_parms = ('alpha_cf', 'alpha_pw', 'alpha_bw')

    for sid in ss.lst_ids:
        fname = model_fname(ss, sid)

        if read_all:
            ch_names, freqs, model_settings, lst_results = \
                read_model(fname)
            read_all = False
        else:
            _, _, _, lst_results = read_model(fname)

        # lst_results is a list of nchan specparam.FitResults objects
        # Extract selected parameters for each channel
        lst_parms = [sid]
        for ich, res in enumerate(lst_results):
            alpha_vals = None   # Dictionary with alpha peak results

            for p in conf['parms']:
                # If any of alpha parms are requested - get all of them:
                if p in alpha_parms:
                    if alpha_vals is None:
                        # This will yield {alpha_parm:val} dictionary
                        alpha_vals = {alpha_parms[i]:val for i,val in \
                                enumerate(get_peak_parms(res, conf['alpha_band']))}

                    lst_parms.append(alpha_vals[p])

                    if make_headers:
                        headers.append(f'{p}{ich}')

                # Add similar "if" clauses here for other expected parameters:
                # ...

        if make_headers:
            # Initialize a DataFrame with using collected headers:
            df = pd.DataFrame(columns=headers)
            make_headers = False

        df.loc[len(df)] = lst_parms

    ss.specparam_df = df
    # print(df)
    print(f'Read specparam data for {len(df)} records')

def get_peak_parms(fitres, band):
    """
    Given specparam.FitResults object and frequency band definition,
    return the central frequency, power and width of a peak (if any) that
    belongs to this band. If several peaks are found, the one with the
    largest amplitude is returned. If no peaks - return NaNs.

    Args:
        fitres (FitResults): FitResults object
        band ([fmin, fmax]): list-like object specifying the band bounds in Hz

    Return:
        cf (float): peak central frequency, Hz or NaN
        pw (float): peak power in log10 units, or NaN
        bw (float): peak width, Hz, or NaN
    """
    peaks = fitres.peak_params      # npeaks x [cf, pw, bw]
    max_pw = -1000.
    cf = peaks[:,0]
    pw = peaks[:,1]
    bw = peaks[:,2]
    ret_cf = np.nan
    ret_bw = np.nan

    for i in range(len(cf)):
        if (cf[i] >= band[0]) and (cf[i] <= band[1]):
            if pw[i] > max_pw:
               ret_cf = cf[i]
               max_pw = pw[i]
               ret_bw = bw[i]

    max_pw = max_pw if max_pw > 0 else np.nan

    return ret_cf, max_pw, ret_bw

def plot_pc_vs_geovar(ss):
    """
    Plot a principal component of a selected parameter versus
    a geo-electromagnetic variable. Requires `read_specparam_data()`
    `read_geoem_data()` to be run first.

    Args:
        ss(obj): reference to this app object

    Returns:

    """
    # Only keep geo data for the records in specparam.df
    # First, get only geo data for records in meta data,
    # which are already filtered to a list of hospital's scan ids
    meta_geo_df = match_meta_to_geoem_data(ss)

    # Now remove records that do not have metadata from the original specparam_df 
    # (specparam_df has data for all patients, while metadata may be just for
    # outpatients)
    df1 = ss.specparam_df   # Just for brevity
    df2 = meta_geo_df
    tmp_df = df1[df1['scan_id'].isin(df2[ss.args['meta_data']['scan_id']])]

    # Now tmp_df is specparam data that corresponds to the records in meta_geo_df
    # However the order of records can in principle be different.
    # So finally reorder rows in tmp_df to match the order of values in meta_geo_df
    specparam_df = tmp_df.set_index('scan_id').loc[df2[ss.args['meta_data']['scan_id']]].reset_index()
    del tmp_df  # specparam_df is a new df, it is safe to delete tmp_df

    conf = ss.args['pc_geoem_scatter']  # Settings for the PCA and scatter plot
    ncomponents = conf['ncomponents']
    parm = conf['parm']                 # This is the specparam to plot

    # The 'parm' vector is nchannel size. Change to PCA basis and take
    # only specified number of components. pca_data is (nrows x ncomponents) np array
    pca_data, variances, ratios = param_to_pca(specparam_df, parm, 
            ncomponents = ncomponents, missing = conf['missing'])

    lst_vars = conf['geovar']       # A list of geo variables to plot against parm
    lst_pc_nums = conf['plot_components']   # A list of PC components to plot
    m = len(lst_pc_nums)
    n = len(lst_vars)
    prefix = parm + ', pc'  # Start of the subplot title

    # squeeze = False ensures that always m x n array object is returned
    # even if it is 1 x 1 array
    fig, axs = plt.subplots(m, n, squeeze = False, figsize=conf['figsize'])

    for j in range(n):
        varname = lst_vars[j]
        y = meta_geo_df[varname].values

        for i in range(m):
            pc_num = lst_pc_nums[i]
            x = pca_data[:,pc_num]
            ax = axs[i,j]
            ax.scatter(x, y, s = conf['marker_size'])
            ax.set_title(f'{prefix}{pc_num} vs {varname}')
            ax.set_xlabel(f'pc{pc_num}')
            ax.set_ylabel(varname)

    plt.tight_layout()

    fname = ss.input_dir + '/' + ss.args['hospital'] + conf['fname_suffix']
    plt.savefig(fname, format='png', dpi=conf['dpi'])

    if not ss.cluster_job:
        plt.show()

def match_meta_to_geoem_data(ss):
    """
    Concatenate meta data with matching geoem data. The matching
    geoem data has timestamp which is earlier than the meta data
    time stamp by the 'lead time' interval. This interval is set in hours
    relative to the meta data time stamp. If such record is not
    found, corresponding metadata record will not be present in the
    merged dataframe.

    The 'lead time' interval described above is specified by a key  
        `ss.args['pc_geoem_scatter']['lead_time']`

    Returned is a dataframe which concatenates meta_data and corresponing
    geo-em data.
    """
    lead_time = pd.Timedelta(hours=ss.args['pc_geoem_scatter']['lead_time'])
    gdf = ss.geoem_df.copy()
    gdf['Datetime'] += lead_time
    meta_geo_df = pd.merge(ss.meta_df, gdf,
                           left_on = 'Datetime',
                           right_on = 'Datetime', 
                           how = 'inner')
    del gdf
    return meta_geo_df

def param_to_pca(df, parm, ncomponents = None, missing = None):
    """
    Convert multi-channel time data for selected parameter to
    specified number of principle components time courses.

    Args:
        df (DataFrame): nrows x ncols data frame containing columns
            with names like 'parmN', where 'parm' is the name of
            the parameter and 'N' is the channel number (0-based).
        parm (str): parameter name (i.e. 'cf').
        ncomponents (int): number of PC components to keep; if None - all
            nchan components will be kept.
        missing (str): strategy for treating missing values. One of
            'mean', 'median' or 'most_frequent'. If not specified, then
            'mean' will be used.

    Returns:
        pca_data (ndarray): shape (nrows, ncomponents) PC time courses for
            the chosen parameter
        variances(ndarray): shape(ncomponents,) variances for each component
        ratios(ndarray): shape(ncomponents,) part of total variance explained by
            each component

    """
    pattern = rf'^{parm}\d+$'
    data = df.filter(regex=pattern).values  # (nrows x nchan) np array
    pca, variances, ratios, _, _ = data_to_pca(data.T, n_components = ncomponents,
                                               missing = missing)

    # pca is (n_components x nrows) np array; we return (nrows x n_components) to be
    # consistent with the original data frame structure
    return pca.T, variances, ratios

def dates_distr(ss):
    """
    Plot histogram of all dates of available records
    """
    date_col = ss.args['meta_data']['scan_date_col']    # Column name for the scan dates
    df = ss.meta_df

    df[date_col] = pd.to_datetime(df[date_col])     # Convert dates to datetime format
    ss.meta_df.set_index(date_col, inplace=True)    # Set dates as df index

    # Count the number of records for each date
    date_counts = df.index.value_counts().sort_index()

    # Plot the histogram
    parms = ss.args['plots']['dates_distr']
    plt.figure(figsize=parms['figsize'])

    # plt.bar(date_counts.index, date_counts.values)
    plt.plot(date_counts.index, date_counts.values, marker='.', linestyle='None')

    plt.xlabel(parms['xlabel'])
    plt.ylabel(parms['ylabel'])
    plt.title(ss.args['hospital'])
    plt.xticks(rotation=parms['xticks_rotation'])

    fname = ss.input_dir + '/' + ss.args['hospital'] + parms['fname_suffix']
    plt.savefig(fname, format='png', dpi=parms['dpi'])

    if not ss.cluster_job:
        plt.show()

def _input_fname(ss, sid):
    '''
    Generate full file name for input power spectrum for given scan ID
    '''
    in_suffix = '_psd.hdf5' if ss.args['what'] == 'sensors' else '_src_psd.hdf5'
    return ss.input_dir + '/' + sid + in_suffix

def model_fname(ss, sid):
    '''
    Generate full pathname for the specparam model fit results for given scan ID
    '''
    suffix = '_psd_model.hdf5' if ss.args['what'] == 'sensors' else '_src_psd_model.hdf5'
    return ss.input_dir + '/' + sid + suffix

def get_data_folders(args):
    '''Setup input and output data folders, metadata and geomagnetic data file
    pathnames depending on the host machine.

    Args:
        args (dict): dictionary with input arguments read from the 
            `INPUT_JSON_FILE`

    Returns:
        data_root (str): path to the root input folder
        out_root (str): path to the root output folder
        meta_csv (str): pathname of the metadata .CSV file
        geoem_dat (str): pathname of the OMNI geomagnetic data file
        cluster_job ( bool): a flag indicating whether the host is on CC cluster
    '''
    what = args['what']

    valid_whats = {'sensors', 'sources'}
    if what not in valid_whats:
        raise ValueError(f'Invalid argument \'{what}\' passed; should be one of {valid_whats}')

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

    meta_csv = args['hosts'][host]['meta_csv']
    geoem_dat = args['hosts'][host]['geoem_dat']

    if host == 'other':
        work_dir = os.getcwd()
        data_root = work_dir + '/' + data_root
        out_root = work_dir + '/' + out_root
        meta_csv = work_dir + '/' + meta_csv
        geoem_dat = work_dir + '/' + geoem_dat

    return data_root, out_root, meta_csv, geoem_dat, cluster_job

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
        'dates_distr': dates_distr,
        'read_geoem_data': read_geoem_data,         # QQQQ
        'read_specparam_data': read_specparam_data, # QQQQ
        'plot_pc_vs_geovar': plot_pc_vs_geovar,
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


