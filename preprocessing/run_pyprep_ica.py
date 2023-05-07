import sys
import glob
import os
import os.path as path
import socket
import mne

from individual_func import write_mne_edf
from  do_pyprep import Pipeline

def get_data_folders():
    '''Setup input and output data folders depending on the host machine.

    Returns:
        data_root, out_root, cluster_job (str, str, bool): paths to the root input and output
        data folders, and a flag indicating whether host is on CC cluster
        
    '''
    host = socket.gethostname()

    # path.expanduser("~") results in /home/<username>
    user_home = path.expanduser("~")
    user = path.basename(user_home) # Yields just <username>

    if 'ub20-04' in host:
        mne.viz.set_browser_backend('matplotlib')
        data_root = '/data/eegfhabrainage/processed'
        out_root = '/data/eegfhabrainage/after-prep-ica'
        cluster_job = False
    elif 'cedar' in host:
        data_root = user_home + '/projects/rpp-doesburg/' + user + '/data/eegfhabrainage/processed'
        out_root = user_home + '/projects/rpp-doesburg/' + user + '/data/eegfhabrainage/after-prep-ica'
        cluster_job = True
    else:
        mne.viz.set_browser_backend('matplotlib')
        home_dir = os.getcwd()
        data_root = home_dir + '/processed'
        out_root = home_dir + '/after-prep-ica'
        cluster_job = False

    return data_root, out_root, cluster_job

if __name__ == '__main__': 
    # ---------- Inputs ------------------
    N_ARRAY_JOBS = 100       # Number of parallel jobs to run on cluster

    #hospital = 'Burnaby'     # Either Burnaby or Abbotsford
    hospital = 'Abbotsford' # Either Burnaby or Abbotsford

    # Abbotsford
    #source_scan_ids = ['1a02dfbb-2d24-411c-ab05-1a0a6fafd1e5']

    """
    # This is a Burnaby subset:
    source_scan_ids = ['2f8ab0f5-08c4-4677-96bc-6d4b48735da2',
                       'fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2',
                       '81be60fc-ed17-4f91-a265-c8a9f1770517',
                       'ffff1021-f5ba-49a9-a588-1c4778fb38d3',
                       '81c0c60a-8fcc-4aae-beed-87931e582c45']
    """

    source_scan_ids = None   # None or a list of specific scan IDs (without .edf)

    view_plots = False       # Flag to show interactive plots (lots of those)
    verbose = 'ERROR'     # Can be 'ERROR', 'CRITICAL', or 'WARNING' (default)
    # ------ end of inputs ---------------

    data_root, out_root, cluster_job = get_data_folders()
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
        source_scan_ids = [path.basename(f)[:-4] for f in glob.glob(input_dir + '/*.edf')]

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

    if source_scan_ids is None:
        scan_files = [path.basename(f) for f in glob.glob(input_dir + '/*.edf')]
    else:
        scan_files = [scan_id + '.edf' for scan_id in source_scan_ids]

    success = True
    for f in scan_files:
        filepath = input_dir + '/' + f

        try:
            # Initiate the preprocessing object
            p = Pipeline(filepath, view_plots = view_plots)
     
            # Apply PREP and ICA
            p.applyPipeline(applyICA = True, view_plots = view_plots)
     
            # Get the resulting mne.Raw object
            raw = p.getRaw()

            # Old version: drop bad channels and save results in 
            # EDF format
            # raw.drop_channels(raw.info['bads'])
            # output_path = output_dir + '/' + f
            # write_mne_edf(raw, fname=output_path, overwrite=True)

            # Keep the bad channels just in case, and save data in .fif file
            output_path = output_dir + '/' + f[:-4] + '_raw.fif'
            raw.save(fname = output_path, proj = False, fmt = 'single', overwrite = True)
            print('\n***** Processing of {} completed\n'.format(f), flush = True)
        except Exception as e:
            success = False
            print('\n***** Record {} !!! FAILED !!!'.format(f))
            print(e, flush = True)
            print('\n')
 
    print("\n{} files processed {}.".format(len(scan_files), \
          'successfully' if success else 'with errors'))

