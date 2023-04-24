# EEG preprocessing pipeline for FHA EDF dataset

## Summary

The code is aimed to preprocess clinical EEG recordings and make them a suitable input for later analyses and ML applications. 
- **Input**: EDF file
- **Output**: 
  - (1) Pandas DataFrame with columns: file name, interval start, interval end, and data - Numpy array of shape (20, 500 * target length) 
  - (2) new EDF file(s) containing only clean EEG interval of target length

**Parameters**:
- number of clean EEG slices to extract from each EDF file
- target length of each slice, in seconds

## Performed operations
1) resample each recording's signal to a common frequency, since some recordings might have different sampling frequencies. 
2) bandpass filter to a specified frequency band to avoid unwatned interference, specifically to exclude the electric power grid frequencies and patient's motion related artifacts
3) identify and remove intervals of special procedures performed on patients during recordings, such as hyperventilation (deep breathing) and photic stimulation (flashing light). Physicians apply these tests to patients in order to detect brain abnormal activity for epilepsy diagnosis. Since these procedures cause abnormal brain activity, and weren't performed for all subjects, we exclude them from the analyses. Also the recordings contain intervals with no signal. This is the results of turned off equipment or disconnected electrodes. So we also have to avoid these flat intervals with zero signal. Thus traget slices acquired only from clean intervals from each EEG, without flat intervals, hyperventilation and photic stimulation. Also a certain time interval at the beginning of the recording is marked as "bad" by default. All types of "bad" intervals are allowed to overlap.
4) **in case of extracting to Numpy arrays** the signal values are also ZScore noramalized. This doesn't apply when saving output to EDF file(s).

All preprocessing parameters may be specified either explicitly, or their default values will be used. The lattter are stored in file **preproc_conf.json** described later in this document.

## Usage
You need both modules edf_preprocessing.py and individual_func.py. The later contains python routine for saving output in EDF format again.

This is an example script which uses class **PreProcessing**. 

```python

from edf_preprocessing import PreProcessing

file_name = "81c0c60a-8fcc-4aae-beed-87931e582c45.edf"
path = "/home/mykolakl/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby/" + file_name
output_path = "your_folder"

# Initiate the preprocessing object, filter the data between 0.5 Hz and 55 Hz and resample to 200 Hz.
p = PreProcessing(path, target_frequency=200, lfreq=0.5, hfreq=55)

# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
p.extract_good(target_length=60, target_segments=5)

# Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
p.save_edf(folder=output_path, filename=file_name)

# Extract and convert data to Numpy arrays
p.create_intervals_data()
df = p.intervals_df

```

**NOTE**
When extracting to Numpy, saving this dataframe to a csv file will damage your data. 
It converts Numpy arrays to string and truncate it, leaving few values in the beginning and the end.
    
To save your data in numpy arrays format for later use, you can use this code:
```python
numpy_data = np.stack(df['data'])

np.save(output_path, numpy_data)
```

As the result, you will get numpy array of shape (number_of_slices, 20, lengths_in_seconds * 500). Following the above example it is (5, 20, 30000).
You should also save inital df to csv to keep track of scan IDs for later match with target label, like age.


## Pipeline for extraction from multiple files

Preprocessing and saving new data to multiple EDF files into target folder

```python
from edf_preprocessing import slice_edfs

import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('warning')

source_folder = "/home/mykolakl/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby"
target_folder = "eeg_fragments_10sec"

labels_file = pd.read_csv('age_ScanID.csv')
scan_ids = labels_file[~(labels_file['AgeYears'].isnull())]['ScanID'] # <- drop records without age

# take scan ids from the list, look for them in the source folder, and apply the preprocessing to at most 100 files;
# filter the data to the default band specified in preproc_conf.json, resample it to 500 Hz (!!! this overrides the default
# setting stored in preproc_conf.json file !!!), extract 1 segment of 10 seconds from each EDF file,
# save new segments as EDFs into the target folder

slice_edfs(source_folder=input_dir, target_folder=output_dir, target_length = 10, source_scan_ids = scan_ids,
           target_frequency=500, target_segments = 1, nfiles=100)
           
# if you don't need files limit - don't specify the parameter "nfiles", default is None

```

## Loading data

```python
from edf_preprocessing import load_edf_data

folder = 'eeg_fragments_10sec'
label_file = 'age_ScanID.csv'

# X - np_array of shape (n_samples, 20, length is seconds * frequency), 
# labels - pd.DataFrame with scan_ids and age, same length as X
X, labels = load_edf_data(folder, label_file)
```

## Preprocessing configuration
All default preprocessing configuration parameters are stored in the JSON file `preproc_conf.json`. Those can
be changed when instantiating `PreProcess` class or calling its methods by passing a custom JSON configuration
file or equivalent Python dictionary object, or by specifying some of the parameters explicitly. The parameters
may be changed in a similar way when processing multiple records with `slice_edfs()` function.

Specifically, an alternative configuration file name is passed to `PreProcess` class constructor (or to `slice_edfs()`)
via an argument `conf_json = <file-pathname>`; an equivalent dictionary object may be passed as `conf_dict = <dictionary-object>`.
A subset of individual configuration parameters may be given explicitly using corresponding keywords -
for example, `target_frequency = 300`.

Note that explicitly supplied parameter values take precedence over those provided in conf_dict; the latter take precedence
over values found in `conf_json` file. 

The meaning of parameters in the JSON configuration file is explained in the comments in the code below. Comment lines start
with the `#` character. Mind that **comments are NOT allowed in a real JSON file** - please remove those if using this example
in practice.

```python
{
	# A list of mandatory EEG channels. An input recording is discarded if any of those is missing
	# (case-insensitive)
	"target_channels":  ["FP1", "FPZ", "FP2", "F3", "F4", "F7", "F8", "FZ", "T3", "T4",
			     "T5",  "T6",  "C3",  "C4", "CZ", "P3", "P4", "PZ", "O1", "O2"],

	# A list of optional non-EEG channels that will be included in the output EDF, if present
	# (case-insensitive)
	"opt_channels":     ["ECG1", "ECG2", "EKG", "EKG1", "EKG2", "EOG 1", "EOG 2", "EOG1", "EOG2", "L EOG",
			     "R EOG","PG1",  "PG2", "A1",   "A2"],

	# Known EOG channel names
	"eog_channels":     ["EOG 1", "EOG 2", "EOG1", "EOG2", "L EOG", "R EOG", "PG1", "PG2"],

	# Known ECG channel names
	"ecg_channels":     ["ECG1", "ECG2", "EKG", "EKG1", "EKG2"],

	# A list of channels that will be removed from the output recording, if present
	# Note that this list is treated as !!! CASE-SENSITIVE  !!! by MNE
	"exclude_channels": ["AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8", "DC1", "DC2",
			     "DC3",  "DC4",  "DIF1", "DIF2", "DIF3", "DIF4",
			     "", "L SPH", "R SPH",
			     "aux1", "aux2", "aux3", "aux4", "aux5", "aux6", "aux7", "aux8", "dc1", "dc2",
			     "dc3",  "dc4",  "dif1", "dif2", "dif3", "dif4",
			     "l sph", "r sph",
                             "Patient Event", "Photic", "Trigger Event", "x1", "x2",
			     "phoic", "Phoic", "photic", 
			     "PATIENT EVENT", "PHOTIC", "TRIGGER EVENT", "X1", "X2",
			     "PHOIC", "PHOTIC"],
	# A list of channels that if encountered, will be renamed to standard 1020 names
	"rename_channels":
		{
			"L EOG": "EOG1",
			"R EOG": "EOG2",
			"EKG": "EKG1"
		},

	# A flag to print out the auxiliary channels included in the output EDF (true or false, lower case)
	"print_opt_channels": false,

	"discard_at_start_seconds": 420,	# time interval removed from the begining of the input record
	"target_frequency":	    256,	# the sampling frequency of the output record
	"target_band":              [0.5, 55],	# the frequency band of the output record
	"target_segments":	    1,		# Max number of good continuous segments to extract
	"target_length":	    360,	# The good segment length in seconds
	"powerline_frq": 	    60.0,	# Power line main frequency, Hz
	"allow_upsampling": 	    false,	# Allow upsampling a record if its sampling rate
						# is smaller than requested

	# Parameters to identify flat intervals
	"flat_parms":
		{
			"flat_max_ptp": 1e-06,	# max amplitude peak-to-peak value for the flat interval
			"bad_percent": 50.0,	# min percentage of the time the channel's peak
            					# to peak is below the 'flat_max_ptp' threshold
						# to be considered flat
			"min_duration": 10.0	# minimum interval in seconds for all consecutive samples to
            					# be below the 'flat_max_ptp' to indicate a flat interval
		},

	"HV_regexp":	"H.*V.*\\d+\\s*[MmIiNn]{3}",	# Regular expression to identify HV annotations like "HV 1 Min"
	"hv_pad_interval": 30,				# Padding interval in seconds around HV series. Final HV boundaries
							# are set as follows:
							# HV start = 1st HV mark - pad_interval
							# HV end = last HV mark + 60 + pad_interval

	"photic_starts": ["Hz"],	# Keyword in annotation that marks the start of the photic stim
	"photic_ends": ["Off"],		# !! EXACT WORDING !! of the annotation that marks the end of the photic stim

	"max_isi": 360,			# Max interval in seconds between photic stimulations to consider those
					# belonging to the same photic stimulation series
	"max_rec_length": 3600		# Max allowed EDF file length in seconds (skip longer files)
}

```

## Running PREP step and ICA artifact removal
The PREP step implements a EEG preprocessing procedure published in the literature. It performs the following steps:
- Powerlines removal

- Re-referencing

- Identifying bad channels

The ICA artifact removal is applied after PREP has successfully completed. It attemts to identify EOG and 
ECG artifacts mixed into sensor channels' waveforms, and remove those.

Please note that during the processing, the EOG and ECG  channels are filtered to different frequency bands
(the actual channel names corresponding to the EOG, ECG channel types may vary -
see the `preproc_conf.json` file above). However, no additional filtering of EEG sensor channels is
performed.

It is assumed that EDF records submitted to the PREP/ICA operations have already passed through the basic preprocessing
stage described above (see section "*Pipeline for extraction from multiple files*").

For an example code demonstrating how PREP and ICA artifact removal are performed please refer to the 
source file `tst_PREP.py`. Parameter values used for PREP/ICA operations are defined in a JSON file
`pyprep_ica_conf.json`, which is described below. Note again that JSON files can not contain comments;
therefore comments in the code below should be removed if one wants to use this code in practice.

```python
{
	"montage": "standard_1020",	# As is
	"powerline_frq": 60.0,		# Powerline frq, Hz

	# Arguments passed to mne.raw.filter() for EOG channels
	"eog_filter_kwargs":
	{
		"l_freq": 1.0, "h_freq": 5.0, 
		"picks": null, "filter_length": "auto", "l_trans_bandwidth": "auto",
		"h_trans_bandwidth": "auto", "n_jobs": null, "method": "fir", "iir_params": null,
		"phase": "zero", "fir_window": "hamming", "fir_design": "firwin",
		"skip_by_annotation": ["edge", "bad_acq_skip"], "pad": "reflect_limited",
		"verbose": null
	},

	# Arguments passed to mne.raw.filter() for ECG channels
	"ecg_filter_kwargs":
	{
		"l_freq": 8.0, "h_freq": 16.0, 
		"picks": null, "filter_length": "auto", "l_trans_bandwidth": "auto",
		"h_trans_bandwidth": "auto", "n_jobs": null, "method": "fir", "iir_params": null,
		"phase": "zero", "fir_window": "hamming", "fir_design": "firwin",
		"skip_by_annotation": ["edge", "bad_acq_skip"], "pad": "reflect_limited",
		"verbose": null
	},

	# Parameter set for the PREP step
	"prep":
	{
		"prep_params":
		{
			"ref_chs": "eeg",
			"reref_chs": "eeg",
			"line_freqs": [60.0], 
			"max_iterations": 4			
		},
		"other_kwargs":
		{
			"ransac": true,
			"channel_wise": false,
			"max_chunk_size": null,
			"random_state": 12345,
			"filter_kwargs":
			{
				"method": "fir"
			},
			"matlab_strict": false
		}
	},

	# Parameter set for the ICA artifact removal step
	"ica":
	{
		"applyICA": true,

		"init":
		{
			"n_components":  0.99999,
			"random_state": 12345,
			"method": "fastica",
			"max_iter": "auto",
			"verbose": null
		},

		"fit":
		{
			"picks": "eeg",
			"tstep": 2.0,
			"verbose": null
		},

		"find_bads_eog":
		{
			"measure": "zscore",
			"threshold": 3.0,
			"verbose": null
		},

		"find_bads_ecg":
		{
			"method": "correlation",
			"measure": "zscore",
			"threshold": 3.0,
			"verbose": null
		}
	},

	# Parameters used for plotting EEG waveforms and spectra
	"plot":
	{
		"time_window": 40.0,
		"scalings": "auto",
		"fmin": 0.0,
		"fmax": 100.0,
                "fstep": 10.0,
		"spect_log_x": true,
		"spect_log_y": true,
		"n_fft": 1024
	}
	
}

```

## Setting up Python virtual environment on Compute Canada cluster
The following steps should be performed to run the code on Compute Canada:
- Create your working folder for the project

- Upload the Python sources and JSON configuration file to your source files location
on the cluster

- Assume that the virtual environment name is **`mne`**. Change your current folder to
the project working folder and perform the following commands:
```
        module load python/3.8.10
        module load scipy-stack

        virtualenv --no-download mne
        source mne/bin/activate
        pip3 install --no-index --upgrade pip
        pip3 install wheel --no-index
        pip3 install mne[hdf5]

        pip3 install pyqt5 --no-index
        pip3 install pyedflib
        pip3 install pandas --no-index
        pip3 install mne-qt-browser     # If one wants to use QT backend
	pip3 install pyprep
	deactivate
```
- In your sbatch scripts, use commands
```
        module load python/3.8.10
        module load scipy-stack
	cd <your working folder>
        source mne/bin/activate

	< run your python program >

	deactivate
```
