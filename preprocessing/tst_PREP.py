import sys
import mne
from mne import viz
from individual_func import write_mne_edf
from  do_pyprep import Pipeline

"""This file takes as input a clean EEG segment and outputs a resampled, filtered signal which does not contain 
bad channels(as per PREP pipeline). It also performs ICA on the given segment, and finally performs current source 
density on the raw data. It outputs the ready to be used EEG signals in raw format."""

# filepath = "/data/eegfhabrainage/srirupa/Clean EEG/clean_data_2.edf"
filepath = "/data/eegfhabrainage/processed/Abbotsford/800c7738-239b-46db-9612-328411369a9d.edf"	# Abbotsford
#filepath = "/data/eegfhabrainage/processed/Abbotsford/fffcedd0-503f-4400-8557-f74b58cff9d9.edf"	# Abbotsford, bad 2 sec at the start
#filepath = "/data/eegfhabrainage/processed/Burnaby/81c0c60a-8fcc-4aae-beed-87931e582c45.edf"	# Burnaby 

output_path = "/data/eegfhabrainage/srirupa/Processed EEG/processed_data_1.edf"
view_plots = True
run_csd = False

mne.viz.set_browser_backend('matplotlib')	# AM: otherwise QT is used for some reason

# Initiate the preprocessing object
p = Pipeline(filepath, view_plots = view_plots)

# Calling the function filters the data between 0.5 Hz and 55 Hz, resamples to 500 Hz
# and performs ICA after applying the PREP pipeline to remove bad channels
p.applyPipeline(applyICA = True, view_plots = view_plots)

# Calling the function gets the pre-processed data in raw format
raw = p.getRaw()

# Calling the function drops the bad channels(as per PREP pipeline)
raw.drop_channels(raw.info['bads'])

# Calling the function saves pre-processed EDF files to output_folder.
write_mne_edf(raw, fname=output_path, overwrite=True)

if not run_csd:
    sys.exit()

# Applies current source density on raw EEG files
raw_csd = mne.preprocessing.compute_current_source_density(raw)

# Plots the current source density of the pre-proceesed EEG
artifact_picks = mne.pick_channels(raw_csd.ch_names, include=raw_csd.ch_names)

if view_plots:
    raw_csd.plot(order=artifact_picks, n_channels=len(artifact_picks),
                    show_scrollbars=False, duration=5, start=0, block=True, 
                    scalings='auto')

# TODO: save Laplace-tansformed EDF

