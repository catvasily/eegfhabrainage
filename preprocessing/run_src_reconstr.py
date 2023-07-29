import sys
import glob
import os
import os.path as path
import socket
import json
import numpy as np
import mne

from nearest_pos_def import nearestPD
from do_src_reconstr import fwd_file_name, construct_noise_and_inv_cov,\
    compute_source_timecourses, get_beam_weights, compute_roi_time_courses,\
    get_voxel_coords, write_roi_time_courses, ltc_file_name, get_label_coms
from add_virtual_channels import add_virtual_channels

JSON_CONFIG_FILE = "src_reconstr_conf.json"
'''Default name (without a path) for the JSON file with parameter settings for
the source reconstruction step. This file is expected
to reside in the same folder as this file.

'''

PYPREP_CONFIG_FILE = "pyprep_ica_conf.json"
'''Name (without a path) for the JSON file with parameter settings for
the PYPREP/ICA step. This file is expected to reside in the same folder
as this script.

'''

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_JSON_CONFIG_PATHNAME = path.dirname(__file__) + "/" + JSON_CONFIG_FILE
_PYPREP_CONFIG_PATHNAME = path.dirname(__file__) + "/" + PYPREP_CONFIG_FILE

sys.path.append(path.dirname(path.dirname(__file__))+ "/misc")
from view_raw_eeg import view_raw_eeg

def get_data_folders():
    '''Setup input and output data folders depending on the host machine.

    Returns:
        data_root (str): path to the root input folder
        out_root (str): path to the root output folder
        fs_dir (str): path to the folder with the template subject 'fsaverage' data
        cluster_job ( bool): a flag indicating whether the host is on CC cluster
    '''
    host = socket.gethostname()

    # path.expanduser("~") results in /home/<username>
    user_home = path.expanduser("~")
    user = path.basename(user_home) # Yields just <username>

    # data_root - where the input comes from; out_root - where the output goes
    if 'ub20-04' in host:
        mne.viz.set_browser_backend('matplotlib')
        data_root = '/data/eegfhabrainage/after-prep-ica'
        out_root = '/data/eegfhabrainage/src-reconstr'
        fs_dir = user_home + '/mne_data/MNE-fsaverage-data'
        cluster_job = False
    elif 'cedar' in host:
        data_root = '/project/6019337/databases/eeg_fha/preprocessed/001_a01_01/'
        out_root = user_home + '/projects/rpp-doesburg/' + user + '/data/eegfhabrainage/src-reconstr'
        fs_dir = user_home + '/projects/rpp-doesburg/' + user + '/data/mne_data/MNE-fsaverage-data'
        cluster_job = True
    else:
        mne.viz.set_browser_backend('matplotlib')
        work_dir = os.getcwd()
        data_root = work_dir + '/processed'
        data_root = work_dir + '/after-prep-ica'
        out_root = work_dir + '/src-reconstr'
        fs_dir = work_dir + '/mne_data/MNE-fsaverage-data'
        cluster_job = False

    return data_root, out_root, fs_dir, cluster_job

if __name__ == '__main__': 
    # ---------- Inputs ------------------
    N_ARRAY_JOBS = 100       # Number of parallel jobs to run on cluster

    #hospital = 'Burnaby'     # Burnaby, Abbotsford, RCH, etc.
    #hospital = 'Abbotsford' 
    hospital = 'RCH' 

    # Abbotsford subset
    #source_scan_ids = ['1a02dfbb-2d24-411c-ab05-1a0a6fafd1e5']

    # Burnaby subset:
    #source_scan_ids = ['2f8ab0f5-08c4-4677-96bc-6d4b48735da2',
    #                   'fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2',
    #                   '81be60fc-ed17-4f91-a265-c8a9f1770517',
    #                   'ffff1021-f5ba-49a9-a588-1c4778fb38d3',
    #                   '81c0c60a-8fcc-4aae-beed-87931e582c45']

    source_scan_ids = None   # None or a list of specific scan IDs (without .edf)

    recalc_forward = False    # Force recalculating forward solutions (even if already exist)
    view_plots = True         # Flag to show interactive plots
    plot_sensors = False      # Flag to plot montage and source space in 3D
    plot_waveforms = False    # Flag to plot sensor and source waveforms

    # Channels to plot if plot_waveforms = True. Set to None to plot all
    # channels
    #plot_chnames = None
    plot_chnames = ['O1','O2','P3','P4','Pz','G_occipital_middle-lh', \
        'G_occipital_middle-rh', 'G_occipital_sup-lh', 'G_occipital_sup-rh', \
        'Pole_occipital-lh', 'Pole_occipital-rh']
    # Scale factor for virtual channels for simultaneously plotting EEG and source
    # waveforms. If the latter are in pseudo-Z units, their magnitudes are typicaly
    # around 1, while EEGs have magnitudes ~1e-5
    vc_scale_factor = 1e-5    # Only affects visual display of the virtual channels

    verbose = 'INFO'     # Can be ‘DEBUG’, ‘INFO', ERROR', 'CRITICAL', or 'WARNING' (default)
    # ------ end of inputs ---------------

    data_root, out_root, fs_dir, cluster_job = get_data_folders()
    input_dir = data_root + "/" + hospital
    output_dir = out_root + "/" + hospital
    fs_subject_dir = fs_dir + "/fsaverage"

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
        # To get bare ID need to chop off "_raw.fif" at the end
        source_scan_ids = [path.basename(f)[:-8] for f in glob.glob(input_dir + '/*.fif')]

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

    scan_files = [scan_id + '_raw.fif' for scan_id in source_scan_ids]

    # Load config
    with open(_JSON_CONFIG_PATHNAME, "r") as fp:
        conf_dict = json.loads(fp.read())

    # Also load pyprep configuration, as we need some data from there
    with open(_PYPREP_CONFIG_PATHNAME, "r") as fp:
        pyprep_dict = json.loads(fp.read())

    # Create montage
    montage_kind = pyprep_dict["montage"]
    montage = mne.channels.make_standard_montage(montage_kind)

    # MRI<->head transformation. In fact, the one stored in the .fif file is head->MRI
    # but MNE funcs will invert it internally if the opposite is needed
    trans_path = path.join(fs_subject_dir, "bem", "fsaverage-trans.fif")

    # Template subject's paths to bem solution and source space;
    src_path = path.join(fs_subject_dir, "bem", conf_dict["source_space"])
    bem_path = path.join(fs_subject_dir, "bem", conf_dict["bem_sol"])

    trans = mne.read_trans(trans_path, verbose=verbose)

    # Read the atlas ROIs ("labels")
    mri_labels = mne.read_labels_from_annot("fsaverage",	    # FreeSurfer subject
                                        parc=conf_dict["parcellation"],       # parcellation (atlas)
                                        hemi='both',                          # 'lh', 'rh' or 'both'
                                        surf_name=conf_dict["surface"],       # which surface:
                                                            # white = white/gray boundary;
                                                            # pial = gray/cereb fluid boundary
                                        annot_fname=None,   # .annot file - instead of 'parc' and 'hemi'
                                        regexp=None,        # regexp to get a subset of labels
                                        subjects_dir=fs_dir,# subjects data dir
                                        sort=True,          # sort labels by name
                                        verbose=verbose)

    src_space = mne.read_source_spaces(src_path, verbose = verbose)

    # Get source space labels with only voxels used as sources
    # - as opposed to dense voxels set of the FreeSurfer 
    labels = [l.restrict(src_space) for l in mri_labels]
    del mri_labels

    # Remove labels (ROIs) that do not have any sources
    labels = [l for l in labels if len(l.vertices)]
    label_names = [label.name for label in labels]
    label_coms = get_label_coms(labels, fs_dir)

    # Settings for source time course reconstructions - see a call to
    # compute_source_timecourses() below
    inverse_method = conf_dict["inverse_method"]    # Inverse solution type

    # Beamformer source reconstruction
    beam_type = conf_dict["beam_type"]              # Beamformer type for method = 'beam'
    src_units = conf_dict["src_units"]
    rcond = 1./conf_dict["max_condition_number"]    # Inverse of max condition number for cov matrix
    tol = conf_dict["noise_upper_bound_tolerance"]  # Accuracy of setting noise cov trace upper bound 
                                                    # so that (data_cov - noise_cov) is a pos def matrix
    beam_kwargs = {"beam_type": beam_type, "units": src_units, "tol": tol,
        "rcond": rcond, "verbose": verbose}

    # Min norm reconstruction:
    # ... add settings here

    stc_args = {'beam': beam_kwargs}    # Add names and args for other inverse solutions
                                        # to this dictionary 
    success = True

    # Main loop over all subjects
    for isubject in range(len(source_scan_ids)):
        f = scan_files[isubject]
        scan_id = source_scan_ids[isubject]
        filepath = input_dir + '/' + f
        subject_output_dir = output_dir + "/" + scan_id

        try:
            if not path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            raw = mne.io.read_raw_fif(filepath, preload=True, verbose = verbose)
            raw.set_montage(montage, on_missing='raise')

            if view_plots and plot_sensors:
                # Plot electrodes positions
                mne.viz.plot_alignment(
                    raw.info,
                    surfaces = 'head',
                    coord_frame = 'mri',
                    src=src_space,                     # Only needed to plot source points
                    eeg=["original", "projected"],    # Show original sensors and projected
                                                      # and projected to the scalp
                    trans=trans,
                    show_axes=True,     # head coords - pink, MRI coords - gray
                    mri_fiducials=True,
                    dig="fiducials",    # which digitization points to show
                    verbose = verbose
                )
                input("Press Enter to continue...")
            
            # Fwd sol calc and raw.get_data('eeg') INCLUDES BAD CHANNELS, so:
            raw = raw.pick('eeg', exclude = 'bads')    # Now raw contains only good EEG channels
                                                       # and no other channels

            eeg_data = raw.get_data(    # eeg_data is nchannels x ntimes
                picks = 'eeg',          # bads are already dropped
                start=0,                # starting time sample number (int)
                stop=None,
                reject_by_annotation=None,
                return_times=False,
                units=None,             # return SI units
                verbose=verbose)

            # Compute forward solutions. Should be done for each subject as the
            # the EEG channels subset actually used does vary. 
            fwd_file = subject_output_dir + "/" + fwd_file_name(scan_id, conf_dict["source_space"]) 

            if recalc_forward or (not path.exists(fwd_file)):
                fwd = mne.make_forward_solution(
                          raw.info, 
                          trans = trans,
                          src = src_space, 
                          bem = bem_path,
                          meg = False,
                          eeg=True,
                          mindist = conf_dict["min_dist_to_skull_mm"],
                          ignore_ref = False,    # this setting does not matter for EEG
                          n_jobs = -1,           # -1 recalcs to the number of available CPU cores
                          verbose = verbose
                          )

                mne.write_forward_solution(fwd_file, fwd, overwrite=True, verbose=verbose)
            else:
                fwd = mne.read_forward_solution(fwd_file, verbose=verbose) 

            # compute_source_timecourses() returns stc, data_cov, W, U, but: only stc
            # is needed if standard MNE funcs are used later to extract ROI time courses;
            # stc is NOT needed if beamformer reconstruction of ROI time courses.
            stc, data_cov, W, _, pz = compute_source_timecourses(raw, fwd,
                method = inverse_method,
                return_stc = False,
                **(stc_args[inverse_method]))

            label_tcs, label_wts = compute_roi_time_courses(
                inv_method=inverse_method,
                labels = labels, fwd = fwd,
                mode = conf_dict["roi_time_course_method"],
                stc = None if inverse_method == 'beam' else stc,
                sensor_data = eeg_data,
                cov = data_cov,
                W = W,
                verbose = verbose)

            # Uncomment below to compare ROI time courses with those found by standard MNE
            # funcs (very slow). One needs stc != None for this.
            '''
            test_tcs = mne.extract_label_time_course(stc, labels, fwd['src'],
                mode=conf_dict["roi_time_course_method"],    # How to extract a time course for ROI
                allow_empty=False,         # Raise exception for empty ROI 
                return_generator=False,    # Return nRoi x nTimes matrix, not a generator
                mri_resolution=False,      # Do not upsample source space
                verbose=verbose)
            print('max diff = {}'.format(np.max(np.abs(label_tcs - test_tcs))))
            '''

            ltc_file = subject_output_dir + "/" + ltc_file_name(scan_id, conf_dict["source_space"]) 
            label_com_rr = get_voxel_coords(fwd['src'], label_coms)    # rr's will be in head coords
            write_roi_time_courses(ltc_file, label_tcs, label_names,
                vertno = label_coms, rr = label_com_rr, W = label_wts, pz = pz)

            # TO DO:
            #    - (optional) implement source reconstruction with dSPM, for comparison

            if view_plots and plot_waveforms:
                add_virtual_channels(raw, label_names, label_com_rr,
                    vc_scale_factor * label_tcs, verbose = verbose)
                view_raw_eeg(raw = raw, picks = plot_chnames)

            print('\n***** Processing of {} completed\n'.format(f), flush = True)
        except Exception as e:
            success = False
            print('\n***** Record {} !!! FAILED !!!'.format(f))
            print(e, flush = True)
            print('\n')
 
    print("\n{} files processed {}.".format(len(scan_files), \
          'successfully' if success else 'with errors'))

