"""
Display alpha-band source power distribution on the cortex surface
"""
import os.path as path
import sys
import numpy as np
import scipy
import mne

from view_inflated_brain_data import view_inflated_brain_data, expand_data_to_rois
sys.path.append(path.dirname(path.dirname(__file__))+ "/preprocessing")
from do_src_reconstr import read_roi_time_courses

user_home = path.expanduser("~")
user = path.basename(user_home) # Yields just <username>

# --- Inputs ------
atlas = "aparc.a2009s"
show_atlas = False
hemi = 'both'
#hemi = 'lh'

#hospital = 'Burnaby'
#subject = '2f8ab0f5-08c4-4677-96bc-6d4b48735da2'
#subject = 'ffff1021-f5ba-49a9-a588-1c4778fb38d3'
#subject = '81c0c60a-8fcc-4aae-beed-87931e582c45'

hospital = 'Abbotsford'
#subject = '1a02dfbb-2d24-411c-ab05-1a0a6fafd1e5'
#subject = '800c7738-239b-46db-9612-328411369a9d'
subject = 'fffcedd0-503f-4400-8557-f74b58cff9d9'

src_data_dir = '/user/' + user + '/data/eegfhabrainage/src-reconstr/' + hospital + '/' + subject
fs_dir = user_home + '/mne_data/MNE-fsaverage-data'
eeg_fif_file = '/user/' + user + '/data/eegfhabrainage/after-prep-ica/' + hospital + '/'\
     + subject + '_raw.fif'
fwd_file = src_data_dir +  '/' + subject + '-ico-3-fwd.fif'
ltc_file = src_data_dir +  '/' + subject + '-ico-3-ltc.hdf5'

expand_values_to_roi = True
band = [8., 12.]   # Frequency band of interest, Hz
n_fft = 512
verbose = 'INFO'
# --- end of inputs ------

raw = mne.io.read_raw_fif(eeg_fif_file, preload=False, verbose=verbose)
fs = raw.info['sfreq']

# Get the source time courses
(label_tcs, label_names, vertno, rr, W, pz) = read_roi_time_courses(ltc_file)
(f, spect)= scipy.signal.welch(label_tcs, fs, nperseg=n_fft, scaling = 'density')
idx=np.logical_and(f>=band[0], f<=band[1])
spect = np.mean(spect[:,idx], axis = 1)    # spect is a 1D array of n_ROI points

if expand_values_to_roi:
    # Create a set of Label objects corresponding to label_names
    atlas_labels = mne.read_labels_from_annot(
        "fsaverage",
        parc = atlas,
        hemi = hemi if hemi != 'split' else 'both',
        surf_name='white', 
        subjects_dir=fs_dir,
        sort=True,                       # Sort labels in alphabetical order
        verbose=verbose
    )

    # atlas_labels may in principle be ordered differently than those in .hdf5 file,
    # and contain extra (unused) labels.
    dd = dict()
    for l in atlas_labels:
        dd[l.name] = l

    # Create a label list in accordance with label_names
    labels = [dd[name] for name in label_names]

    # Expand ROI single value to the whole ROI
    spect, vertno = expand_data_to_rois(spect, labels)

brain = view_inflated_brain_data(
        atlas = atlas,
        show_atlas = True,
        hemi = hemi,
        title = None,
        data = spect,
        cbar_lims = None,
        colormap = 'auto',
        #alpha_data = 0.25,
        alpha_data = 1.,
        smoothing_steps = None,
        rois_to_mark = None,
        vertno = vertno,
        show_vertices = False,
        subjects_dir = fs_dir,
        scale_factor = 0.2,
        color_dots = 'white',
        alpha_cortex = 1.,
        resolution = 50,
        show = True,
        block = False,
        inflated = True,
        kwargs_brain = None,
        kwargs_data = None,
        verbose = verbose
    )

input("Press ENTER to continue...")

# Save a screenshot
brain.save_image("screenshot.png")

