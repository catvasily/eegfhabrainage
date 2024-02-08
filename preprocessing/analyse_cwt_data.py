"""
**A script to analyse CWT results**  

For each record, do some value checks on the estimated Exponentiated Weibull
(EW) distribution parameters. Specifically, identify ch, frq pairs where ac < 1, which means
that PSD is no longer finite at 0. 

Calculate parameter means and stds for all (ch, frq) pairs over all records in the hospital,
and save them as separate TFData files.
"""
import sys
import os.path as path
import glob
import commentjson as cjson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_welch import get_data_folders
from tfdata import TFData

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

sys.path.append(path.dirname(path.dirname(__file__))+ "/misc")
from running_mean_std import RunningMeanSTD

INPUT_JSON_FILE = "analyse_cwt_input.json"  # This script input parameters
'''Name (without a path) for the JSON file with input arguments 
for individual runs of the CWT step
'''

def main():
    # Parse the input JSON that was used to genereate CWTs
    with open(pathname(INPUT_JSON_FILE), 'r') as fp:
        args = cjson.loads(fp.read())

    what = args['what']                     # 'sensors' or 'sources'
    hospital = args['hospital']             # Burnaby, Abbotsford, RCH, Surrey

    # Scan ID to print out the results, or None
    print_id = args['print_id']

    # Channel name for printing single channel results
    print_chan = args['print_chan']

    # Frequency value for printing single frequency results
    print_freq = args['print_freq']

    cwt_root, cluster_job = get_data_folders(args)[1:]    # Root folder for CWT results - all hospitals
    cwt_dir = cwt_root + "/" + hospital     # hospital subfolder path

    # Flag to show (ch, f) scatter plot for pairs with ac < 1
    show_scatter_plot = False if cluster_job else args['show_scatter_plot']

    # Func to map the scan_id to a full CWT file path
    cwt_pathname = lambda scan_id:   \
            cwt_dir + '/' + scan_id + \
            ('_tfd.hdf5' if what == 'sensors' else  '_src_tfd.hdf5')

    # Define the set of records to process
    source_scan_ids = args['source_scan_ids']   # None or a list of specific scan IDs (without .edf)

    # Run through all the records only if requested
    if not args['print_only']:
        if source_scan_ids is None:
            if what == 'sensors':
                # To get bare ID need to chop off '_tfd.hdf5' at the end
                source_scan_ids = [path.basename(f)[:-9] for f in glob.glob(cwt_dir + '/*.hdf5')]
            else:
                # To get bare ID need to chop off '_src_tfd.hdf5' at the end
                source_scan_ids = [path.basename(f)[:-13] for f in glob.glob(cwt_dir + '/*.hdf5')]

        df = None
        summator = RunningMeanSTD()

        for scan_id in source_scan_ids:
            if scan_id[:len(hospital)] == hospital: # Skip records with already collected stats
                continue

            tfd = TFData.read(cwt_pathname(scan_id))
            df = check_cwt_parms(tfd, df)       # Collect (ch, frq) cases with unusual parm values
            summator.push(tfd.data.to_numpy())

        # Print abnormal parms found
        if df is not None:
            print('NOTE: condition ac < 1 was found violated in some cases')
            #print(df.to_string())

            scatter_plot(df, cwt_dir, args, show_scatter_plot)

            with pd.option_context('display.max_rows', None,
                'display.max_columns', None,
                'display.width', None,
                'display.precision', 3,):
                print(df)

        # Save stats if not plot_only
        if not args['plot_only']:
            scan_id = hospital + '_mean'
            tfd_mean = TFData(shape = tfd.data.shape, scan_id = scan_id,
                    values = summator.mean(), ch_names = tfd.ch_names,
                    freqs = tfd.freqs, parm_names = tfd.parm_names)

            tfd_mean.write(cwt_pathname(scan_id))

            scan_id = hospital + '_std'
            tfd_std = TFData(shape = tfd.data.shape, scan_id = scan_id,
                    values = summator.std(), ch_names = tfd.ch_names,
                    freqs = tfd.freqs, parm_names = tfd.parm_names)

            tfd_std.write(cwt_pathname(scan_id))

    # This will print any record results; to print overall stats
    # set "<hospital>_<mean or std>" as print_id in INPUT_JSON_FILE
    if print_id is not None:
        print(f'\nStat results for fit parameters for {print_id}:\n')
        if print_chan is not None:
            df = TFData.read(cwt_pathname(print_id)).to_pandas(chan = print_chan)[0]    # nf x nparms
            print(f'\nAmplitude distribution parameters for channel {print_chan}')
            print(df)

        if print_freq is not None:
            df, factual = TFData.read(cwt_pathname(print_id)).to_pandas(freq = print_freq)    # nf x nparms
            print(f'\nAmplitude distribution parameters for frequency {factual:.1f} Hz')
            with pd.option_context('display.max_rows', None,
                'display.max_columns', None,
                'display.width', None,
                'display.precision', 3,):
                print(df)

def check_cwt_parms(tfd, append_to = None):
    """
    Run sanity checks on CWT amps distribution parameters for each channel
    and each frequency of given TFData. Currently only condition ac < 1 is
    checked; after uncommenting parts of the code may also check for a < 1
    and the location shift ew_loc > 0. 

    Args:
        tfd (TFData): the CWT amps EW distribution parameters for one record
        append_to (DataFrame): a dataframe to append the results to

    Returns:
        df (DataFrame): a dataframe listing scan ID, channel, frequency, location,
            a, c, ac where ac<1 is encountered. If `append_to` dataframe was provided,
            then it will be updated with the new data, if not - it will be
            returned unchanged. If append_to = None and no new data was found,
            None is returned.
    """
    array = tfd.data.to_numpy()

    """
    # Check if r <= 0. Result: sometimes >0, but then it is usually small (1e-2)
    iloc = tfd.parm_names.index('ew_loc')

    # Get an array of (ich, ifreq) with positive loc shifts
    idx_positive = np.argwhere(array[:,:,iloc] > 0)

    if idx_positive.size:
        print(f'Positive location shifts found for rec {tfd.scan_id}:')
        for ich, ifreq in idx_positive:
            print('ch {}, frq {:.1f}: ew_loc = {:.1e}'.format(tfd.ch_names[ich],
                tfd.freqs[ifreq], array[ich, ifreq, iloc]))
    """

    iloc = tfd.parm_names.index('ew_loc')
    ia = tfd.parm_names.index('ew_a')
    ic = tfd.parm_names.index('ew_c')

    """
    # Check if a >=1. Result: no, for low freqs (1 - 3 Hz) it may be 0.7 - 0.9
    idx_bad_a = np.argwhere(array[:,:,ia] < 1)

    # Check if 
    if idx_bad_a.size:
        print(f'Values a < 1 found for rec {tfd.scan_id}:')
        for ich, ifreq in idx_bad_a:
            print('ch {}, frq {:.1f}: ew_a = {:.1e}'.format(tfd.ch_names[ich],
                tfd.freqs[ifreq], array[ich, ifreq, ia]))
    """

    # Verify that a*c > 1. Otherwise the PDF has a singularity at x = ew_loc
    idx_bad_ac = np.argwhere(array[:,:,ia]*array[:,:,ic] < 1)

    if idx_bad_ac.size:
        columns = ['Scan ID', 'ich', 'ch', 'f', 'loc', 'a', 'c', 'a*c']

        data = list()
        for ich, ifreq in idx_bad_ac:
            data.append([tfd.scan_id, ich, tfd.ch_names[ich], tfd.freqs[ifreq],
                array[ich, ifreq, iloc],
                array[ich, ifreq, ia], array[ich, ifreq, ic],
                array[ich, ifreq, ia]* array[ich, ifreq, ic]])

        df = pd.DataFrame(data, columns = columns)

        if append_to is not None:
            return pd.concat([append_to, df], ignore_index = True)

        return df

    return append_to
            
def scatter_plot(df, outdir, args, show_plot):
    """
    Display a scatter plot of (ch#, f) pairs where ac<1.

    Args:
        df (DataFrame): a dataframe returned by `check_cwt_parms()`
            with fields `scan ID, channel, frequency, location,
            a, c, ac` (where ac<1)            
        outdir (str): pathname of the folder to save the plot PNG
        args (dict): a dictionary of arguments for this program; should
            contain a key 'scatter_plot' with plotting args
        show_plot (bool): flag to display the interactive plot. If `False`, 
            the plot won't be shown but still saved to the file.

    Returns:
        axs (Axes): the matplotlib `Axes` object which carries the scatter plot
    """
    dfc = df.groupby(['ich','f'])['Scan ID'].count().reset_index(name = 'cnt')
    kwargs = args['scatter_plot']
    kwargs['title'] = args['hospital'] + ': ' + kwargs['title']
    kwargs['xlim'][1] = df['ich'].max()
    kwargs['ylim'][1] = df['f'].max()

    dpi = kwargs['dpi']
    del kwargs['dpi']

    axs = dfc[['ich', 'f','cnt']].plot.scatter(x='ich', y='f', c = 'cnt', colormap='jet', **kwargs)
    fig = plt.gcf()
    plt.savefig(outdir + '/ch_f_ac_less_than_1.png', dpi=dpi, format='png')

    if show_plot:
        plt.show()

    return axs

# ---- Epilogue -----------------------------
if __name__ == '__main__': 
    main()

