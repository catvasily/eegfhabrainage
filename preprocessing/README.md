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
A subset of individual configuration paramters may be given explicitly using corresponding keywords -
for example, `target_frequency = 300`.

Note that explicitly supplied parameter values take precedence over those provided in conf_dict; the latter take precedence
over the parameter values read from `conf_json` file. 

The meaning of parameters in the JSON configuration file is explained in the comments in the code below, which start
with `#` character. Mind that **comments are NOT allowed in a real JSON file** - please remove those if using this example
in practice.

```python
{
	# A list of mandatory channels. An input recording is discarded if any of those is missing
	"target_channels":  ["FP1", "FPZ",   "FP2",  "F3",   "F4",   "F7",     "F8",   "FZ",   "T3",   "T4",
			     "T5",  "T6",    "C3",   "C4",   "CZ",   "P3",     "P4",   "PZ",   "O1",   "O2"],

	# A list of channels that will be removed from the input recording, if present
	"exclude_channels": ["A1",   "A2",   "AUX1", "AUX2", "AUX3", "AUX4",   "AUX5", "AUX6", "AUX7", "AUX8",
			     "Cz",   "DC1",  "DC2",  "DC3",  "DC4",  "DIF1",   "DIF2", "DIF3", "DIF4", "ECG1",
			     "ECG2", "EKG1", "EKG2", "EOG 1","EOG 2","EOG1",   "EOG2", "Fp1",  "Fp2",  "Fpz", 
			     "Fz",   "PG1",  "PG2",  "Patient Event", "Photic", "Pz", "Trigger Event", "X1", "X2", "aux1",
			     "phoic", "photic"],

	"discard_at_start_seconds": 420,	# time interval removed from the begining of the input record
	"target_frequency":	    200,	# the sampling frequency of the output record
	"target_band":              [0.5, 55],	# the frequency band of the output record

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

	"hyperventilation":
		{
			# Sets of keywords in the input record annotations that identify beginnings and ends
			# of the hyperventilation intervals
			"hv_1min_start_notes": ["HV 1Min", "HV 1 Min"],
			"hv_1min_end_notes":   ["Post HV 30 Sec", "Post HV 60 Sec", "Post HV 90 Sec"],
			"hv_start_notes":      ["HV Begin", "Begin HV"],
			"hv_end_notes":        ["HV End", "End HV"],

			# Additional padding in seconds added before and after the hyperventilation interval
			# boundaries obtained from annotations
			"hv_pad_interval": 90
		},

	# Keywords in the input record annotations that identify photic stimulation
	"photic_stim_keywords": ["Hz"]
}

```

