'''
**A top level script for running the source reconstruction step.**
'''
import sys
import glob
import os
import os.path as path
import socket
import commentjson as cjson
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

def get_data_folders(args):
    '''Setup input and output data folders depending on the host machine.

    Args:
        args (dict): dictionary with input arguments read from the 
            `INPUT_JSON_FILE`

    Returns:
        data_root (str): path to the root input folder
        out_root (str): path to the root output folder
        fs_dir (str): path to the folder with the template subject 'fsaverage' data
        cluster_job ( bool): a flag indicating whether the host is on CC cluster
    '''
    host_found = False
    host = _get_host(args)

    for key in args['hosts']:
        if key in host:
            host = key
            host_found = True
            break

    if not host_found:
        host = 'other'

    cluster_job = args['hosts'][host].get('cluster_job', False)

    if not cluster_job:
        mne.viz.set_browser_backend('matplotlib')

    # Get the host data
    # data_root - where the input comes from; out_root - where the output goes
    data_root = args['hosts'][host]['data_root']
    out_root = args['hosts'][host]['out_root']
    fs_dir = args['hosts'][host]['fs_dir']

    if host == 'other':
        work_dir = os.getcwd()
        data_root = work_dir + '/' + data_root
        out_root = work_dir + '/' + out_root
        fs_dir = work_dir + '/' + fs_dir

    return data_root, out_root, fs_dir, cluster_job


def _normalize_hospitals(hospital_cfg):
    if isinstance(hospital_cfg, str):
        hospitals = [hospital_cfg]
    elif isinstance(hospital_cfg, list) and all(isinstance(h, str) for h in hospital_cfg):
        hospitals = hospital_cfg
    else:
        raise ValueError('"hospital" must be either a string or a list of strings in src_reconstr_conf.json')

    if not hospitals:
        raise ValueError('"hospital" list in src_reconstr_conf.json should not be empty')

    return hospitals


def _validate_source_scan_ids(source_scan_ids, hospitals):
    if source_scan_ids is None:
        return

    if not isinstance(source_scan_ids, list) or not all(isinstance(s, str) for s in source_scan_ids):
        raise ValueError('"source_scan_ids" must be null or a list of strings in src_reconstr_conf.json')

    if len(hospitals) != 1:
        raise AssertionError('When "source_scan_ids" is provided, exactly one hospital must be specified.')

if __name__ == '__main__': 
    # Load config
    with open(_JSON_CONFIG_PATHNAME, "r") as fp:
        cfg = cjson.loads(fp.read())

    # Also load pyprep configuration, as we need some data from there
    with open(_PYPREP_CONFIG_PATHNAME, "r") as fp:
        pyprep_dict = cjson.loads(fp.read())

    # ---------- Inputs ------------------
    N_ARRAY_JOBS = cfg['N_ARRAY_JOBS']  # Number of parallel jobs to run on cluster

    hospitals = _normalize_hospitals(cfg['hospital'])
    source_scan_ids = cfg['source_scan_ids']
    _validate_source_scan_ids(source_scan_ids, hospitals)
    view_plots = cfg['view_plots']
    verbose = cfg['verbose']    # Can be ‘DEBUG’, ‘INFO', ERROR', 'CRITICAL', or 'WARNING' (default)
    # ------ end of inputs ---------------

    data_root, out_root, fs_dir, cluster_job = get_data_folders(cfg)
    fs_subject_dir = fs_dir + "/fsaverage"

    mne.set_log_level(verbose=verbose)

    # When running on the CC cluster, 1st command line argument is a 0-based
    # array job index
    if len(sys.argv) == 1:   # No command line args
        ijob = 0
    else:
        ijob = int(sys.argv[1])

    # Create montage
    montage_kind = pyprep_dict["montage"]
    montage = mne.channels.make_standard_montage(montage_kind)

    # MRI<->head transformation. In fact, the one stored in the .fif file is head->MRI
    # but MNE funcs will invert it internally if the opposite is needed
    trans_path = path.join(fs_subject_dir, "bem", "fsaverage-trans.fif")

    # Template subject's paths to bem solution and source space;
    src_path = path.join(fs_subject_dir, "bem", cfg["source_space"])
    bem_path = path.join(fs_subject_dir, "bem", cfg["bem_sol"])

    trans = mne.read_trans(trans_path, verbose=verbose)

    # Read the atlas ROIs ("labels")
    mri_labels = mne.read_labels_from_annot("fsaverage",        # FreeSurfer subject
                                        parc=cfg["parcellation"],       # parcellation (atlas)
                                        hemi='both',                          # 'lh', 'rh' or 'both'
                                        surf_name=cfg["surface"],       # which surface:
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
    # or marked to drop
    labels = [l for l in labels if (len(l.vertices) and \
                    (l.name not in cfg['drop_labels'])) ]
    label_names = [label.name for label in labels]

    print(f'\nTotal of {len(labels)} labels (ROIs) will be processed\n')

    # Uncomment this to print all the labels
    #print(f'Labels, n_vertices for {cfg["parcellation"]} atlas:')
    #for l in labels:
    #    print(l.name, len(l.vertices))

    label_coms = get_label_coms(labels, fs_dir)

    # Settings for source time course reconstructions - see a call to
    # compute_source_timecourses() below
    inverse_method = cfg["inverse_method"]    # Inverse solution type

    # Beamformer source reconstruction
    beam_type = cfg["beam_type"]              # Beamformer type for method = 'beam'
    src_units = cfg["src_units"]
    rcond = 1./cfg["max_condition_number"]    # Inverse of max condition number for cov matrix
    tol = cfg["noise_upper_bound_tolerance"]  # Accuracy of setting noise cov trace upper bound 
                                                    # so that (data_cov - noise_cov) is a pos def matrix
    beam_kwargs = {"beam_type": beam_type, "units": src_units, "tol": tol,
        "rcond": rcond, "verbose": verbose}

    # Min norm reconstruction:
    # ... add settings here

    stc_args = {'beam': beam_kwargs}    # Add names and args for other inverse solutions
                                        # to this dictionary 
    success = True

    for hospital in hospitals:
        input_dir = data_root + "/" + hospital
        output_dir = out_root + "/" + hospital

        if not path.exists(output_dir):
            os.makedirs(output_dir)

        if source_scan_ids is None:
            # To get bare ID need to chop off "_raw.fif" at the end
            hospital_scan_ids = [path.basename(f)[:-8] for f in glob.glob(input_dir + '/*.fif')]
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

        scan_files = [scan_id + '_raw.fif' for scan_id in hospital_scan_ids]

        # Main loop over all subjects
        for isubject in range(len(hospital_scan_ids)):
            f = scan_files[isubject]
            scan_id = hospital_scan_ids[isubject]
            filepath = input_dir + '/' + f
            subject_output_dir = output_dir + "/" + scan_id

            try:
                if not path.exists(subject_output_dir):
                    os.makedirs(subject_output_dir)

                raw = mne.io.read_raw_fif(filepath, preload=True, verbose = verbose)
                raw.set_montage(montage, on_missing='raise')

                if view_plots and cfg['plot_sensors']:
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
                fwd_file = subject_output_dir + "/" + fwd_file_name(scan_id, cfg["source_space"]) 

                if cfg['recalc_forward'] or (not path.exists(fwd_file)):
                    fwd = mne.make_forward_solution(
                              raw.info, 
                              trans = trans,
                              src = src_space, 
                              bem = bem_path,
                              meg = False,
                              eeg=True,
                              mindist = cfg["min_dist_to_skull_mm"],
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
                    mode = cfg["roi_time_course_method"],
                    stc = None if inverse_method == 'beam' else stc,
                    sensor_data = eeg_data,
                    cov = data_cov,
                    W = W,
                    verbose = verbose)

                # Uncomment below to compare ROI time courses with those found by standard MNE
                # funcs (very slow). One needs stc != None for this.
                '''
                test_tcs = mne.extract_label_time_course(stc, labels, fwd['src'],
                    mode=cfg["roi_time_course_method"],    # How to extract a time course for ROI
                    allow_empty=False,         # Raise exception for empty ROI 
                    return_generator=False,    # Return nRoi x nTimes matrix, not a generator
                    mri_resolution=False,      # Do not upsample source space
                    verbose=verbose)
                print('max diff = {}'.format(np.max(np.abs(label_tcs - test_tcs))))
                '''

                ltc_file = subject_output_dir + "/" + ltc_file_name(scan_id, cfg["source_space"]) 
                label_com_rr = get_voxel_coords(fwd['src'], label_coms)    # rr's will be in head coords

                # Save EEG sensor time courses rather than label_tcs, to save disk space
                # (for MEG that would not make much sense)
                write_roi_time_courses(ltc_file, eeg_data, label_names,
                    vertno = label_coms, rr = label_com_rr, W = label_wts, pz = pz,
                    is_sensor_data = True)

                # TO DO:
                #    - (optional) implement source reconstruction with dSPM, for comparison

                if view_plots and cfg['plot_waveforms']:
                    add_virtual_channels(raw, label_names, label_com_rr,
                        cfg['vc_scale_factor'] * label_tcs, verbose = verbose)
                    view_raw_eeg(raw = raw, picks = cfg['plot_chnames'])

                print('\n***** Processing of {} completed\n'.format(f), flush = True)
            except Exception as e:
                success = False
                print('\n***** Record {} !!! FAILED !!!'.format(f))
                print(e, flush = True)
            print('\n')
 
    print("\n{} files processed {}.".format(len(scan_files), \
          'successfully' if success else 'with errors'))

