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

if __name__ == '__main__':	# Need this to avoid running the code during doc generation
    # --- Inputs ------
    atlas = "aparc.a2009s"
    show_atlas = False
    hemi = 'both'
    #hemi = 'lh'
    #hemi = 'split'
    
    hospital = 'Burnaby'
    subject = '2f8ab0f5-08c4-4677-96bc-6d4b48735da2'	# most alpha occip; mostly left hemi
    #subject = 'fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2'
    #subject = '81be60fc-ed17-4f91-a265-c8a9f1770517'
    #subject = 'ffff1021-f5ba-49a9-a588-1c4778fb38d3'
    #subject = '81c0c60a-8fcc-4aae-beed-87931e582c45'
    #subject = '57ea2fa1-66f1-43f9-aa17-981909e3dc96'	# Alpha - FRONTAL!
    
    #hospital = 'Abbotsford'
    #subject = '1a02dfbb-2d24-411c-ab05-1a0a6fafd1e5'
    #subject = '800c7738-239b-46db-9612-328411369a9d'	# Wide spread mostly posterior alpha
    #subject = '800dd0f6-fabb-40c8-afda-3bdaecd55855'
    #subject = 'fffcedd0-503f-4400-8557-f74b58cff9d9'
    
    #hospital = 'RCH'
    #subject = '7cd8be05-5286-4d65-bb5d-42031f131db8'
    #subject = '80cbae5d-625b-473b-bf17-f5352cbf3c0c'	# Wide spread posterior alpha
    #subject = '799d4cae-40e0-449e-9119-793c91b3305c'	# Wide spread LH posterior alpha
    #subject = 'f96b27fa-df7d-4853-86d2-2c57d685c511'	# More or less how it should be
    #subject = 'fc34d5f0-dd6a-4e65-8f87-95ef5b452681'	# Wide spread LH superior alpha
    
    src_data_dir = '/user/' + user + '/data/eegfhabrainage/src-reconstr/' + hospital + '/' + subject
    fs_dir = user_home + '/mne_data/MNE-fsaverage-data'
    eeg_fif_file = '/user/' + user + '/data/eegfhabrainage/after-prep-ica/' + hospital + '/'\
         + subject + '_raw.fif'
    fwd_file = src_data_dir +  '/' + subject + '-ico-3-fwd.fif'
    ltc_file = src_data_dir +  '/' + subject + '-ico-3-ltc.hdf5'
    
    expand_values_to_roi = True
    band = [8., 12.]   # Frequency band of interest, Hz
    n_fft = 512
    cbar_lims = [0.02, 0.04]	# Set to cbar_lims = None for auto
    #cbar_lims = None 
    colorbar = False
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

    kwargs_data = None if colorbar else {"colorbar": False} 
    
    brain = view_inflated_brain_data(
            atlas = atlas,
            show_atlas = True,
            hemi = hemi,
            title = None,
            data = spect,
            cbar_lims = cbar_lims,
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
            kwargs_data = kwargs_data,
            verbose = verbose
        )
    
    brain.show_view(view='lateral', hemi='lh')
    input("Lateral LH. Press ENTER to continue...")
    brain.save_image("lat_lh.png")
    
    brain.show_view(view='lateral', hemi='rh')
    input("Lateral RH. Press ENTER to continue...")
    brain.save_image("lat_rh.png")
    
    brain.show_view(view='rostral', hemi='lh')
    input("Rostal. Press ENTER to continue...")
    brain.save_image("rostal.png")
    
    brain.show_view(view='caudal', hemi='rh')
    input("Codal. Press ENTER to continue...")
    brain.save_image("codal.png")
    
    #brain.show_view(view='dorcal', hemi='rh')	# This one crashes
    #input("Dorcal. Press ENTER to continue...")	# Use axial instead
    
    brain.show_view(view='axial', hemi='rh')
    input("Axial. Press ENTER to continue...")
    brain.save_image("axial.png")
    
    brain.show_view(view='ventral', hemi='rh')
    input("Ventral. Press ENTER to continue...")
    brain.save_image("ventral.png")

    brain.show_view(view='frontal', hemi='lh')
    input("Frontal LH. Press ENTER to continue...")
    brain.save_image("front_lh.png")
    
    brain.show_view(view='frontal', hemi='rh')
    input("Frontal RH. Press ENTER to continue...")
    brain.save_image("front_rh.png")
    
    brain.show_view(view='parietal', hemi='lh')
    input("Parietal LH. Press ENTER to continue...")
    brain.save_image("parietal_lh.png")
    
    brain.show_view(view='parietal', hemi='rh')
    input("Parietal RH. Press ENTER to continue...")
    brain.save_image("parietal_rh.png")
    
    # Save an arbitrary screenshot
    brain.save_image("screenshot.png")

