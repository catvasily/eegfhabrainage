"""
**A top level script for calculating the  Continuous Wavelet Transforms (CWTs)
for sensor and source reconstructed data, and fitting the exponentiated Weibull
distributions to CWT spectral amplitude statistics.**
"""
import sys
import os
import os.path as path
import glob
import commentjson as cjson
import numpy as np
import h5py         # Needed to save/load files in .hdf5 format
import matplotlib.pyplot as plt
from scipy import signal, stats
import warnings
import multiprocessing as mp
from multiprocessing import Pool
import mne

from run_welch import get_data_folders, load_data
from do_src_reconstr import read_roi_time_courses
from tfdata import TFData

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

# These are only needed for benchmarking
sys.path.append(path.dirname(path.dirname(__file__))+ "/misc")
from utils import timeit

PREPROC_CONFIG_FILE = "preproc_conf.json"
'''Name (without a path) for the JSON file with preprocessing parameters 
which was used to run filtering/segmentation step

'''

CWT_CONFIG_FILE = 'cwt_conf.json'   # This script general configuration settings
'''Name (without a path) for the JSON file with configuration parameters 
for the CWT step

'''

INPUT_JSON_FILE = "cwt_input.json"  # This script input parameters
'''Name (without a path) for the JSON file with input arguments 
for individual runs of the CWT step

'''

# Complex Morlet' wavelet 'omega' parameter. This means that the oscillating
# exponent of the wavelet for a scale "s" is given by the expression
#   E(n) = exp(j*OMEGA0*n/s).
PI2 = 2.* np.pi
OMEGA0 = PI2 

# The distribution to fit to spectral amps stats
DIST = None
'''Name of a statistical distribution to be fit to CWT amps distribution - as
defined by "fit_distribution" key in `CWT_CONFIG_FILE`

'''
FITS = None     # fit results; this variable will be set later
'''Global instance of `FitEW` class that stores the distribution fit results

'''

def fit_res(amps, **parms):
    ''' Just a wrapper for FITS(amps, parm) '''
    return FITS(amps, parms['parm'])

# Functions to use for parameter estimation
ESTIMATORS = {'mean': np.mean, 'median': np.median, 'std': np.std,
        'skew': stats.skew, 'kurtosis': stats.kurtosis,
        'ew_a': fit_res, 'ew_c': fit_res, 'ew_loc': fit_res, 'ew_scale': fit_res,
            'ew_fit_stat': fit_res, 'ew_fit_pval': fit_res}

def main():
    global DIST

    # Parse input args
    with open(pathname(INPUT_JSON_FILE), 'r') as fp:
        args = cjson.loads(fp.read())

    N_ARRAY_JOBS = args['N_ARRAY_JOBS']     # Number of parallel jobs to run on cluster
    what = args['what']                     # 'sensors' or 'sources'
    hospital = args['hospital']             # Burnaby, Abbotsford, RCH, Surrey
    source_scan_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf)
    view_plots = args['view_plots']         # Flag to show interactive plots
    plot_only = args['plot_only']           # if true, plot already precalculated spectra

    # IDs of channels to plot if view_plots = True, or [] or None
    plot_ids = args['plot_ids']

    # Channels/ROIs to plot, or None
    plot_chnames = args['plot_chnames'][what]

    # Frequencies for distribution plots, or None
    plot_freqs = args['plot_freqs']

    # X, Y limits for distribution plots, or None
    plot_xlim = args['plot_xlim'][what]
    plot_ylim = args['plot_ylim'][what]

    # Number of x points for the distribution plots
    plot_nx = args["plot_nx"]

    # Single scan ID to print out the results, or None
    print_id = args['print_id']

    # Channel name for printing single channel results
    print_chan = args['print_chan']

    # Frequency value for printing single frequency results
    print_freq = args['print_freq']

    verbose = args['verbose']               # Can be ‘DEBUG’, ‘INFO', ERROR', 'CRITICAL', or 'WARNING' (default)
    data_root, out_root, cluster_job = get_data_folders(args)
    # ------ end of args parsing ---------------

    input_dir = data_root + "/" + hospital
    output_dir = out_root + "/" + hospital
    
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    mne.set_log_level(verbose=verbose)

    # When running on the CC cluster, 1st command line argument is a 0-based
    # array job index
    if len(sys.argv) == 1:   # No command line args
        ijob = 0
    else:
        ijob = int(sys.argv[1])

    if source_scan_ids is None:
        if what == 'sensors':
            # To get bare ID need to chop off "_raw.fif" at the end
            source_scan_ids = [path.basename(f)[:-8] for f in glob.glob(input_dir + '/*.fif')]
        else:
            # To get bare ID one needs to get folders with names that are 5 HEX numbers
            # separated by 4 dashes. Poor man's solution for it is just '*-*-*-*-*'
            source_scan_ids = [path.basename(f) for f in glob.glob(input_dir + '/*-*-*-*-*')]

    if cluster_job:
        view_plots = False    # Disable interactive plots, just in case
        nfiles = len(source_scan_ids)
        files_per_job = nfiles // N_ARRAY_JOBS + 1
        istart = ijob * files_per_job

        if istart > nfiles - 1:
            print('All done')
            sys.exit()

        iend = min(istart + files_per_job, nfiles)
        source_scan_ids = source_scan_ids[istart:iend]

    if what == 'sensors':
        scan_files = [scan_id + '_raw.fif' for scan_id in source_scan_ids]
    else:
        scan_files = [scan_id + '/' + scan_id + '-ico-3-ltc.hdf5' \
                for scan_id in source_scan_ids]

    # load the original preprocessing configuration, as we need some data from there
    with open(pathname(PREPROC_CONFIG_FILE), "r") as fp:
        preproc_dict = cjson.loads(fp.read())

    # --- Set up this script's general configuration parameters ---
    with open(pathname(CWT_CONFIG_FILE), "r") as fp:
        conf_dict = cjson.loads(fp.read())

    fs = preproc_dict['target_frequency']   # Sampling frequency
    frange = conf_dict['freq_range']        # Frequency range
    nf =  conf_dict['freq_points']          # Number of log-spaced frequency points (including fmin, fmax)

    parm_dict = conf_dict['spect_parms']
    nparms = len(parm_dict)

    DIST = getattr(stats, conf_dict['fit_distribution'])
    # ---- end of setup ----------

    # Get the frequencies
    freqs = loglin_freqs(frange, nf, fs)
    print('freqs:', freqs)

    # Wavelet central frequency for scale s = 1: fc = (OMEGA0/2 pi)*fs
    # Time t = n/fs, then in real time units oscillating part of the wavelet is
    #   E(t) = exp(j*OMEGA0*n/s) = exp(j*OMEGA0*(fs/s)*t)
    # Thus the central frequency (in Hz) of the wavelet at scale s is:
    #   fc(s) = OMEGA0*fs/s/(2 pi)
    fc0 = OMEGA0*fs/PI2      # Central frequency for scale s = 1

    # For scale s, we have fc(s) = fc0/s, therefore:
    scales = fc0/freqs

    outname = lambda scan_id:   \
            output_dir + '/' + scan_id + \
            ('_tfd.hdf5' if what == 'sensors' else  '_src_tfd.hdf5')

    if not plot_only:
        # --- main loop ----
        if cluster_job:
            ncpus=int(os.environ['SLURM_CPUS_PER_TASK'])  # get the number of cpus allocated
        else:
            ncpus = max(int(mp.cpu_count()/2), 1)

        print('\nProcessing record IDs:')
        for scan_id, f in zip(source_scan_ids, scan_files):
            print(f'{scan_id}')
            data, ch_names = load_data(input_dir + '/' + f)     # Data is nchans x ntimes
            #data = data[:,:400]      # QQQ Otherwise distr fit runs forever
            nchans = len(ch_names)

            z = stats.zscore(data, axis = 1, **conf_dict['zscore'])

            for ich, ch in enumerate(data):
                if np.allclose(ch, 0):
                    warnings.warn(f'Flat channel "{ch_names[ich]}" was found in record "{scan_id}".')
                    z[ich] = 0.

            # Create (nchans x nf x nparms) xarray to store CWT results
            tfd = TFData(shape = (nchans, nf, nparms), scan_id = scan_id,
                        values = None, ch_names = ch_names, freqs = freqs, parm_names = list(parm_dict.keys()))

            # scipy's cwt() only processes 1 channel at a time, so
            results = list()    # multiprocessing AsyncResult objects of each channel

            with Pool(processes = ncpus) as pool:
                for ich in range(nchans):
                    # Calculate the wavelet transform
                    # tfd.data[ich,:,:] = cwt_process_a_channel(z[ich], scales, parm_dict)
                    results.append(pool.apply_async(cwt_process_a_channel, (z[ich], scales, parm_dict)))

                for ich in range(nchans):
                    tfd.data[ich,:,:] = results[ich].get()

            # Save the results                    
            tfd.write(outname(scan_id))
            print('Done\n')
            #break   # QQQ run one subject only
            # ------ end of main loop ---------------

        print('All records processed\n')

    if view_plots and (plot_ids is not None):
        for pid in plot_ids:
            title = f'ID = {pid}.'
            plot_distr(outname(pid), plot_chnames, freqs = plot_freqs, title = title, 
                    xlim = plot_xlim, ylim = plot_ylim, nx = plot_nx)

        plt.show()

    if print_id is not None:
        if print_chan is not None:
            df = TFData.read(outname(print_id)).to_pandas(chan = print_chan)[0]    # nf x nparms
            print(f'\nAmplitude distribution parameters for channel {print_chan}')
            print(df)

        if print_freq is not None:
            df, factual = TFData.read(outname(print_id)).to_pandas(freq = print_freq)    # nf x nparms
            print(f'\nAmplitude distribution parameters for frequency {factual:.1f} Hz')
            print(df)

    # ----- end main -------------------------

def cwt_process_a_channel(z, scales, parm_dict):
    '''
    Perform all CWT-related processing for a single channel.

    Args:
        z (ndarray): `shape (ntimes,)` - single channel z-score timecourse
        scales(ndarray): `shape (nf,)` -  a set of scales for CWT
        parm_dict (dict): `conf_dict["spect_parms"]`

    Returns:
        parm_values (ndarray): `shape (nf, nparms)` estimated CWT parameters for all frequencies
    '''
    # Calculate the wavelet transform
    cwt = signal.cwt(z, signal.morlet2, scales, w = OMEGA0)  # nf x ntimes 'complex128' array

    """
    # THE SNIPPET BELOW SHOULD BE COMMENTED OUT. This code is only used to visually compare
    # PSDs obtained with CWT to those obtained by a traditional Welch method.    
    from welch_logf import welch_logf

    fmin  = 1.
    fmax = 55.
    nf = 50
    fs = 256.
    freqs = loglin_freqs([fmin, fmax], nf, fs)

    pwr_sp = np.absolute(cwt)**2
    spect_sp = np.mean(pwr_sp, axis = 1)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    axes[0].semilogy(freqs, spect_sp)
    axes[0].set_title('CWT')
    axes[0].set_xticklabels([])

    # Calculate Welch spectrum on a log grid
    fl, Pl, nfft_w = welch_logf(z, fmin, fmax, nf = nf, fs=fs, nperseg = None,
        scaling='density')

    axes[1].semilogy(fl, Pl)
    axes[1].set_title('Welch on a log scale')
    plt.show()
    """

    return estimate_parms(cwt, parm_dict)   # nf x nparms array of floats

class FitEW:
    '''
    An object to calculate and store the fit results of exponentiated Weibul distribution
    for all frequencies
    '''
    ew_parms = {'ew_a':0, 'ew_c':1, 'ew_loc':2, 'ew_scale':3,
            'ew_fit_stat':4, 'ew_fit_pval':5}

    def __init__(self, amps):
        '''
        Args:
            amps(ndarray): shape (nf, ntimes), float > 0 - spectral amplitudes
        '''
        nf = amps.shape[0]
        self.params = np.zeros((nf, len(FitEW.ew_parms)))

        # NOTE. When calling DIST.fit() sometimes 'RuntimeWarning: invalid
        # value encountered in add' occurs. This is due to scipy calculating log(0), getting
        # '-inf' then trying to add it to the float number. The result is NaN; it is expected
        # to occasionally occur by the fit() function and is processed propery (by testing
        # it with np.isfinite()). Therefore, we disable this warning here.
        # Similarly, other warnings originate from the same stat package
        warnings.filterwarnings('ignore', message='invalid value encountered in add',
            category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='invalid value encountered in divide',
            category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='overflow encountered in scalar multiply',
            category=RuntimeWarning)

        for i in range(nf):
            try:
                params = DIST.fit(amps[i], method='mle')                
                self.params[i,:-2] = params 
                frozen_dist = DIST(*params)
                err = stats.cramervonmises(amps[i], frozen_dist.cdf)    # Fit quality estimate
                self.params[i,-2:] = err.statistic, err.pvalue
            except Exception as e:
                print('FitEW:__init__() error:', e)
                print(f'i (frequency #) = {i}')
            
        warnings.resetwarnings()

    def __call__(self, amps, parm):
        '''
        Returns a vector of specified distribution parameter values
        for all frequencies. `amps` argument is ignored
        '''
        if parm not in FitEW.ew_parms:
            raise ValueError(f'Invalid parameter value \`{parm}\` specified')

        return self.params[:, FitEW.ew_parms[parm]]

#@timeit
def estimate_parms(cwt, parm_dict):
    ''' For each frequency line, calculate a set of parameters of the
    spectral amplitudes distribution. The names of those parameters
    are listed in the `cwt_conf.json` file under key "spect_parms".

    Args:
        cwt (array): nf x ntimes complex array of wavelet transformation coefficients.
        parm_dict (dictionary): keys = parameter names, values = kwargs for corresponding
            function

    Returns:
        parm_values (array): nf x nparms real value of estimated parameters
    '''
    global FITS

    amps = np.abs(cwt)
    nf, namps = amps.shape
    nparms = len(parm_dict)

    values = np.empty((nf, nparms))
    values[:] = np.nan

    # Run the fit for all frequencies
    FITS = FitEW(amps)

    for ip, parm in enumerate(parm_dict.keys()):
        if parm not in ESTIMATORS:
            raise ValueError(f'Parameter to estimate: {parm} - is not known')

        values[:, ip] = ESTIMATORS[parm](amps, **parm_dict[parm])

    return values 

def loglin_freqs(frange, nf, fs):
    '''Calculate a set of frequencies with equally spaced log(f)

    Args:
        frange (list): `[fmin, fmax]` target frequency range, Hz
        nf (int): number of frequency points; should be > 1
        fs (float): sampling frequency, Hz

    Returns:
        fl (nparray): shape (nf,) - list of log-spaced frequencies, Hz
    '''
    fmin, fmax = frange

    if (fmin >= fs/2) or (fmax > fs/2):
        raise ValueError('fmin must be less, and fmax must be less or equal to fs/2')

    if (fmin <= 0) or (fmax <=0) or np.isclose(fmin, fmax):
        raise ValueError('Both fmin and fmax must be positive and not equal')

    if nf <= 1:
        raise ValueError('nf should be larger than 1')

    if fs <= 0:
        raise ValueError('fs should be positive')

    if fmin > fmax:
        tmp = fmax
        fmax = fmin
        fmin = tmp

    lfmin, lfmax = np.log10([fmin, fmax])

    # Get the step in log scale, and equally spaced log10s 
    lf, dlf = np.linspace(lfmin, lfmax, nf, endpoint = True, retstep = True)

    # Target log-spaced frequencies:
    fl = 10**lf
    return fl

def plot_distr(fname, ch_names, freqs = None, title = None, xlim = None, ylim = None, nx = 200):
    '''Plot amplitude distributions for specified channels for a set of
    channels. Exponential Weibul distribution is assumed.

    Args:
        fname (str): full pathname to the .hdf5 file created by
            `TFData.write()`
        ch_names (list of str): list of channel or ROI names to plot
        freqs (list of floats or None): Frequencies to plot, Hz. If not specified
            all frequencies will be plotted. The frequency will always be adjusted
            to the closest ones available in the data.
        title (str or None): plot title
        xlim ([xmin, xmax] or None): X-axis limits
        ylim ([ymin, ymax] or None): Y-axis limits 
        nx (int): number of x-points to plot in PDF(x)

    Returns:
        Nothing
    '''
    PPF_MIN = 0.00  # Min PPF value (for finding x_min)
    PPF_MAX = 0.90  # Max PPF value (for finding x_max)

    tfd = TFData.read(fname)    # nchans x nf x nparms

    if freqs is None:
        freqs = list(tfd.data.coords[TFData.FRQ_DIM])
    else:
        # Adjust frequencies if necessary
        freqs = [tfd.get_nearest_freq(f) for f in freqs]

    parms = list(tfd.data.coords[TFData.PARM_DIM])   # parm names

    istart = parms.index('kurtosis') + 1    # The specific distribution parameters start
                                            # after 'kurtosis' in the parms list

    if xlim is None:
        # Find x limits based on the distribution of the first channel and the
        # first (the lowest) frequency
        # MIND that with label slicing the bounds are INCLUSIVE! That's why + 3, not + 4!
        dist_parms = tfd.data.loc[ch_names[0], freqs[0], parms[istart]:parms[istart + 3]].data # dist_parms itself is xarray again
        xlim = [DIST.ppf(PPF_MIN, *dist_parms), DIST.ppf(PPF_MAX, *dist_parms)]

    x = np.linspace(xlim[0], xlim[1], nx)

    for ch in ch_names:
        plt.figure()

        for f in freqs:
            dist_parms = tfd.data.loc[ch, f, parms[istart]:parms[istart + 3]].data
            plt.plot(x, DIST.pdf(x, *dist_parms))

        plt.title(ch if title is None else title + f' Ch {ch}')
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        if ch == ch_names[0]:
            flabels = [f'{f:.1f}' for f in freqs]
            plt.legend(flabels)

# ---- Epilogue -----------------------------
if __name__ == '__main__': 
    main()

