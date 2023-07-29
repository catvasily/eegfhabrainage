# EEG preprocessing pipeline for FHA EDF dataset

## Summary

The code is aimed to preprocess clinical EEG recordings and make them a suitable input for later analyses and ML applications. 
Code uses the original EEG records in EDF format as input. The following steps (tasks) may be performed:

1. **Filtering, resampling and extracting of good data segments** of a target length. The output is the good segments in EDF format.
2. **Performing EEG PREP procedure and artifact removal**. The input is typically good segments obtained on the 1st step,
    and the output is the records in **.fif** format.
3. **Extracting hyperventilation intervals from the original records**. The extracted intervals (segments) are saved in .EDF format.

Either of these tasks requires specifying numerous processing parameters. To separate the code and parameter data, default values
of parameters are stored in JSON configuration files **`preproc_conf.json`** and **`pyprep_ica_conf.json`**; the latter
is needed only for step 2. These files are expected to reside in the same folder as the top level Python script files.

Key functions and classes by default import JSON configuration files to set most parameter values. One can always override the defaults
a) by specifying their own versions of JSON files as arguments to a class or a function call, b) by providing an equivalent Python dictionary, or
(in some cases) c) by specifying parameters explicitly in a function call. If for some parameter all three options are used at the same time,
explicitly specified values take precedence, then those coming from the Python dictionary, then those coming from the JSON file.

More details about each step and configurations files are given below.

## 1. Filtering, resampling and extracting of good data segments
### Running the code
This step is executed by a top level script **`run_filtering_segmentation.py`**. It calls functions and classes defined in
`edf_preprocessing.py` and `individual_func.py`. Most of the logic is encapsulated in class `PreProcessing`; for practical application
of this class see source code for function `slice_edfs()`. One can find references and full description of both in the 
auto-generated documentation.

When executing the code locally, use this command in Linux terminal:
```
python run_filtering_segmentation.py
```
If running as an array job on the cedar cluster, use this command in your sbatch script (see `eeg_array_job.sbatch` for an example):
```
python run_filtering_segmentation.py ${SLURM_ARRAY_TASK_ID}
```

**Before running**, file **`run_filtering_segmentation.py`** may need to be **modified** by the user as follows.

* **Function `get_data_folders()`** identifies the host computer where the script is executed, and returns `data_root`, `out_root` paths and
a `cluster_job` flag. **`data_root`** points to the top folder where the input
EDF files are located. One level down are subfolders corresponding to each hospital (like `Abbotsford`, `Burnaby`); the EDF records themselves are
located inside the hospital subfolders. Please refer to the file structure in `/project/6019337/databases/eeg_fha/release_001/edf` on cedar as an example. 
The **`out_root`** defines location where the resulting (processed) records are put and has the same structure. ***User may need to add/edit the host definitions
and the returned `data_root`, `out_root` paths as appropriate***, by modifying the following code segment:
```
	if 'ub20-04' in host:
		data_root = '/data/eegfhabrainage'
		out_root = data_root + '/processed'
		cluster_job = False
	elif 'cedar' in host:
		data_root = '/project/6019337/databases/eeg_fha/release_001/edf'
		out_root = user_home + '/projects/rpp-doesburg/' + user + '/data/eegfhabrainage/processed'
		cluster_job = True
	else:
		home_dir = os.getcwd()
		data_root = home_dir
		out_root = home_dir + '/processed'
		cluster_job = False
```

* **User needs to specify which hospital is being processed**, by setting the **`hospital`** variable in the main function of the script:
```
	# Inputs
	...
	hospital = 'Abbotsford'
	...
```

* **When running as an array job on the cluster**, the variable **`N_ARRAY_JOBS`** should be consistent with the  
  **`"--array"`** parameter value in the sbatch script:  
> File **`run_filtering_segmentation.py`**:
```
	# Inputs
	...
	N_ARRAY_JOBS = 100	# Number of parallel jobs to run on cluster
	...
```
>> The **sbatch script** (see **`eeg_array_job.sbatch`** as an example):
```
	...
	#SBATCH --array=0-99	# the last job index should be equal to N_ARRAY_JOBS - 1 
	...
```
* **When only some records for the hospital need to be processed**, provide a list of record IDs in the variable
**`source_scan_ids`**:
```
	# Inputs
	...
	# Use source_scan_ids = None to process all records
	source_scan_ids = ["1a02dfbb-2d24-411c-ab05-1a0a6fafd1e5", "fffaab93-e908-4b93-a021-ab580e573585"]
	...
```

### Performed operations
The following operations are performed for each input record.
* The EDF file is loaded using `mne.io.read_raw_edf()` function, except for the channels specified in the
`exclude_channels` list. This list, as well as other parameters mentioned here, are read from the `preproc_conf.json`
file. The record will not be processed if it does not have all channels listed in `target_channels`, or if it is
too long: longer than `max_rec_length`.
* Some of the channels may be renamed to ensure they are treated correctly by the MNE Python software - see `rename_channels` key in the 
`preproc_conf.json`.
* Based on the channel names, **channel types** are assigned as EEG sensor channels, EOG channels or ECG channels. Channels that
do not belong to either of the above categories are assigned a `'misc'` (miscellaneous) type.
* All EEG, EOG and ECG channels are notch-filtered at power line frequencies
* All EEG channels are additionally band-pass filtered to a `target_band`
* All channels are resampled to a sampling frequency equal to `target_frequency`
* Good segments of a `target_length` are extracted. This involves identifying bad segments first. The bad segments may include:   
    - Flat (no signal) intervals  
    - Periods when photic (optical) stimulation was delivered  
    - Hyper ventilation (HV) periods

  Photic stim and HV intervals are padded at the ends by corresponding `xxx_pad_interval` amount of seconds. A starting
  segment at the beginning of each record is also automatically marked as bad, as specified by the key `discard_at_start_seconds`.
* If a good segment is found, it is saved as an EDF file to the location determined by the `out_root` variable and the
hospital name. The output EDF record preserves the channel type information as well as the filtering parameters
used.  

### JSON configuration file
All default preprocessing configuration parameters are stored in the JSON file `preproc_conf.json`. Default parameter
values can be changed when instantiating the `PreProcess` class or while calling its methods, by passing a custom JSON configuration
file or an equivalent Python dictionary object. Also some of the parameters may be given explicitly as arguments. The same approach applies
to the `slice_edfs()` and some other top level functions.

Specifically, an alternative configuration file name is passed to `PreProcess` class constructor (or to `slice_edfs()`)
via an argument `conf_json = <file-pathname>`; an equivalent dictionary object may be passed as `conf_dict = <dictionary-object>`.
In some cases a subset of individual configuration parameters may be given explicitly using corresponding keywords -
for example, `target_frequency = 300`.

Note that explicitly supplied parameter values take precedence over those provided in `conf_dict`; the latter take precedence
over values found in `conf_json` file. 

Detailed description of all arguments of each function or method are available in the generated documentation.

The meaning of parameters in the JSON configuration file is explained in the comments in the code below. Comment lines start
with the `#` character. IMPORTANTLY, please mind that **comments are NOT allowed in real JSON files**. Please remove them if
using this example JSON snippet in practice.

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
	"HV_end":	"END HV",			# Hyperventilation end annotation, if present
	"hv_pad_interval": 30,				# Padding interval in seconds around HV series. Final HV boundaries
							# are set as follows:
							# HV start = 1st HV mark - 60 - pad_interval
							# HV end = last HV mark + 60 + pad_interval

	"photic_starts": ["Hz"],	# Keyword in annotation that marks the start of the photic stim
	"photic_ends": ["Off"],		# !! EXACT WORDING !! of the annotation that marks the end of the photic stim
	"photic_pad_interval": 30,	# Padding inteval in seconds around photic stim series

	"max_isi": 360,			# Max interval in seconds between photic stimulations to consider those
					# belonging to the same photic stimulation series
	"max_rec_length": 3600		# Max allowed EDF file length in seconds (skip longer files)
}

```

## 2. EEG PREP procedure and artifact removal
### Running the code
This step is done by a top level script **`run_pyprep_ica.py`**. Most of related functions and classes are defined in
the source file `do_pyprep.py`. The main work horse is class `Pipeline`, which in turn uses class `PrepPipeline` imported from a `pyprep` library.
See more details in the autogenerated documentation.

When executing the code locally, use this command in Linux terminal:
```
python run_pyprep_ica.py
```
If running as an array job on the cedar cluster, use this command in your sbatch script (see `eeg_array_job.sbatch` for an example):
```
python run_pyprep_ica.py ${SLURM_ARRAY_TASK_ID}
```

The main script **`run_pyprep_ica.py`** may need to be **modified** by the user, to set the root input and output folders, hospital name, etc.
Please refer to section ["Running the code"](#running-the-code) under the segmentation task
["1. Filtering, resampling and extracting of good data segments"](#1-filtering-resampling-and-extracting-of-good-data-segments), because the
procedure is identical. Note that typically the PREP step is applied to extracted good segments rather than to the original data. 

### Performed operations
* The **EEG PREP step** executes the "PREP" procedure published in the literature as implemented by the `pyprep` library.
It performs the following operations:
    - Powerlines removal (just in case - if the original raw data is supplied on input)
    - Re-referencing
    - Identifying bad channels

* The **ICA artifact removal** is applied after the PREP has completed successfully. This is done using the MNE Python `ICA` class and its methods.
  The procedure attemts to identify EOG and ECG artifacts mixed into the EEG sensor channels using ECG, EOG channel signals as templates, and tries
  to remove them.

  Importantly, during the ICA processing **the EOG and 1st ECG channels are filtered to different frequency bands** and will be stored like that
  in the output.

  The actual channel names of the EOG, ECG channels used may vary from hospital to hospital - see channel type lists in `preproc_conf.json` file.
  However, **no additional filtering of the EEG sensor channels is done**.
* The processed records are **saved in .fif file format**.

### JSON configuration file
Parameter values specific to the PREP/ICA operations are defined in a JSON configuration file
`pyprep_ica_conf.json`, which is described below. Note again that JSON files can not contain comments;
therefore comments in the code below should be removed if one wants to use it in practice.

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

## 3. Extracting hyperventilation intervals from the original records
This step is completely independent from steps 1, 2 and is only used to identify and store the HV segments, as the name suggests.

### Running the code
The top level script to execute is **`extract_hv_intervals.py`**. Internally it calls the same function `slice_edfs()` as in the
[first step](#1-filtering-resampling-and-extracting-of-good-data-segments).

When executing the code locally, use this command in Linux terminal:
```
python extract_hv_intervals.py
```
If running as an array job on the cedar cluster, use this command in your sbatch script (see `eeg_array_job.sbatch` for an example):
```
python extract_hv_intervals.py ${SLURM_ARRAY_TASK_ID}
```

As usual, the main script **`extract_hv_intervals.py`** may need to be **modified** to set the root input and output folders, etc.
Please refer to section ["Running the code"](#running-the-code) under the segmentation task
["1. Filtering, resampling and extracting of good data segments"](#1-filtering-resampling-and-extracting-of-good-data-segments), for details.
Note that this step should only be applied to the original EDF records. 

### Performed operations
* First, the same **basic preprocessing operations as in the task 1 (segmentation) are done**, namely
    * The EDF file is loaded, except for the channels specified in the `exclude_channels` list.
    * Some of the channels may be renamed to ensure they are treated correctly by the MNE Python software
    * Based on the channel names, channel types are assigned.
    * All EEG, EOG and ECG channels are notch-filtered at power line frequencies
    * All EEG channels are additionally band-pass filtered to a `target_band`
    * All channels are resampled to a sampling frequency equal to the `target_frequency`

* **Hyperventilation intervals** identified by markers (annotations) *"HV XX min"* **are extracted**.
  No padding of the HV segments is applied.

* The resulting records are saved in EDF format.

### JSON configuration file
All HV-related configuration parameters are stored in the JSON file `preproc_conf.json`.

## 4. Performing source reconstruction
In this step, we reconstruct the source time courses from a designated set of Regions of Interest (ROIs).
It's important to note that in the MNE Python software, **ROI**s are referred to as "*labeled brain locations*"
or simply "**labels**". Therefore, in this document, we use both terms interchangeably to convey the same meaning.

### Running the code
To perform this step one needs to run a top level script **`run_src_reconstr.py`**. This script depends on source
files `do_src_reconstr.py`, `nearest_pos_def.py` and file `construct_single_source_weights.py` from `beam-python`
repository imported as a git submodule. 

When executing the code locally, run this command in Linux terminal from the `.../eegfhabrainage/preprocessing`
folder:
```
python run_src_reconstr.py
```

If running as an array job on the cedar cluster, use this command in your sbatch script
(see `eeg_array_job.sbatch` for an example):
```
python run_src_reconstr.py ${SLURM_ARRAY_TASK_ID}
```

Typically, the user will need to modify the main script **`run_src_reconstr.py`** as per their requirements.
This involves setting the root input and output folders by modifying function `get_data_folders()`, the 
hospital name, and other relevant parameters which are found at the top of the main function following the line
`if __name__ == '__main__'`. The procedure is similar to the one described in the section 
["Running the code"](#running-the-code) under the segmentation task
["1. Filtering, resampling and extracting of good data segments"](#1-filtering-resampling-and-extracting-of-good-data-segments).

It is important to note that this step expects the input files to be in **`.fif`** format. Normally, the records
obtained after the PREP/ICA step are used as inputs.

### Performed operations
To perform the source reconstruction, the subject's MRI data and the EEG sensor locations on the subject's head
are required. As those are not available in our case, a template MRI for the `fsaverage` subject which comes
with the `FreeSurfer` software, and standard sensor locations for a given montage are utilized.

When using MNE Python one can install the `fsaverage` data by invoking the
[fetch_fsaverage()](https://mne.tools/stable/generated/mne.datasets.fetch_fsaverage.html) function. This only
needs to be done once. Please mind to record the location of the downloaded data as it is being used by
the `get_data_folders()` function mentioned above.  
The downloaded `fsaverage` data includes pre-calculated BEM models and fine grid source spaces suitable for
analyses of high density EEG and MEG recordings. However given that our EEGs have only 20 or fewer
channels, the spatial resolution of our source reconstruction is relatively poor. Consequently, we can utilize
lower density source spaces for the `fsaverage` subject. This choice helps improve computational speed and
memory efficiency. Specifically, we employ a low-density source space named `fsaverage-ico-3-src.fif` with
only around twelve hundred sources, which can be generated by executing a script `make_fsaverage_bem.py`
(this script only needs to be run once).

The `fsaverage` template also comes with two standard brain parcellations which define ROIs (labels) on
the cortical surface. One needs to specify the parcellation to use in the `src_reconstr_conf.json`
configuration file described below. A lower resolution parcellation with 34 ROIs per hemisphere is used
by setting `"parcellation": "aparc"`. A higher resolution parcellation with 74 ROIs per hemisphere will
be utilized with setting: `"parcellation": "aparc.a2009s"` (default).

The following operations are performed in this step.
* The labels (parcellation data), BEM model and the source space are read from the designated `fsaverage`
  data folder.

* Then for each subject:
    * The input EEG record is read and the good EEG sensor channels and their corresponding sensor
      positions are determined.
    * Forward solutions for 3 orthogonally oriented dipoles in each source locations are calculated,
      unless a .fif file with already precalculated forward solutions for this subject (say, from
      previous runs) is found in the output location.
    * Spatial filter weights `W` for each source are constructed using a scalar single source
      minimum variance beamformer. These weights allow finding a time course `s[i](t)` for each
      source using the expression `s[i](t) = W[i]'*b(t)`, where `b(t)` is a vector of EEG
      sensor time courses, and "`'`" denotes vector/matrix transposition. By default, the weights
      are scaled so that reconstructed time course reflects signal's "pseudo-Z" -- that is,
      the original source amplitude divided by the projected noise. Please refer to the sections
      ["Beamformer weights calculation"](#beamformer-weights-calculation) and
      ["Normalizing for group analyses"](#normalizing-for-group-analyses)
      regarding **important details** about the beamformer calculations and scaling of
      the reconstructed source time courses.
    * For each ROI (label), a single time course is created based on the time courses of all
      the sources belonging to that ROI. This operation is currently performed using a PCA
      approach, which is equivalent to `"pca_flip"` mode as defined in MNE package. For details,
      please refer to the documentation for the function
      [extract_label_time_course()](https://mne.tools/stable/generated/mne.extract_label_time_course.html).
      Note that our code uses a different algorithm which gets the same results orders of magnitude
      faster than the MNE implementation.
    * All reconstructed **label time courses** are saved in a **`.hdf5`** file. Additionally
      this file contains: the **ROI names**, the **vertex numbers** corresponding to the **ROI centers of
      mass (COMs)** on the FreeSurfer cortex/white matter boundary surface, **3D locations** of the 
      ROI COMs in the **head** coordinates, **label beamformer weights** `W[l]` and the sensor-level
      **pseudo-Z** of the data.   
      The vertex numbers are encoded so that negative ones refer to the left hemisphere, and 
      non-negative - to the right hemisphere; see function `get_label_coms()` in `do_src_reconstr.py`
      for details.   
      The weights `W[l]` are similar to the single source beamformer weights,
      and enable reconstructing the ROI (label) time course `l[t]` via an expression `l(t) = W[l]'*b(t)`.
      The sensor-level pseudo-Z is defined as `pz = trace(R)/trace(N)`, where `R` and `N` are
      the EEG data and the noise covariance, respectively (see section ["Beamformer weights
      calculation"](#beamformer-weights-calculation) for details).  
      The `.hdf5` file togehter with the subject's forward solution `.fif` file are stored in a
      subfolder named after the subject's scan ID, within the output location.

### Beamformer weights calculation
Beamformer weights are calculated using functions from `construct_single_source_weights.py`
source file which should be located in the `../beam-python` folder relative to this file's
location.

The calculation of the scalar single-source spatial filter begins with determining the source
orientation vector `u` as the initial step. Note that the sign of `u` is ambiguous and may be
randomly assigned by the beamformer software. In order to address this, we select the sign of
`u` in such a way that the angle between `u` and the outward normal to the cortical surface at the
source location is less than 180 degrees.

For each source in the source space, corresponding beamformer weight `w` is calculated
using the following expressions:  
`w = C R^-1 h; h = H*u`  
Here, `R^-1` denotes the *pseudo-inverse* of the sensor covariance matrix. The variable `h`represents
a "scalar" source forward solution, while `H=[h_x, h_y, h_z]` is a triplet of forward solutions for
dipoles oriented along the coordinate axes at the source location. Finally, `C` is a scaling
factor which can be chosen as follows.

To obtain source time courses in physical units corresponding to current dipoles (i.e. `A*m`)
factor `C` should be chosen as  
`C = (h' R^-1 h)^-1`  
This scaling is selected by setting `"src_units": "source"` in the `src_reconstr_conf.json`
configuration file.

However, this approach may lead to deep sources appearing to have disproportionately large
amplitudes due to the small magnitudes of their corresponding forward solutions `h`. 
An alternative option is to obtain source time courses in pseudo-Z units, which involves
normalizing the physical signal amplitudes by the root mean square (RMS) of the noise
projected by the filter to the source location. This scaling is selected by setting
`"src_units": "pz"` in the `src_reconstr_conf.json` file. Such normalization removes the
bias in estimated source magnitudes for deep sources. In this case the scaling factor
is determined by the formula  
`C = 1/sqrt(h' * R^-1 * N * R^-1 h)`   
where `N` represents the *noise covariance* matrix. The latter is found as described
below.

In the case of resting state activity, directly measuring the noise covariance `N `is
not possible because, by definition, there are no 'control' intervals without the
presence of the 'activity of interest'. To address this issue, various approaches
have been suggested in the literature. Here we calculate the noise covariance under
the assumption that the noise field represents an *'uninformed prior,'* meaning the
lack of any specific prior information about brain sources magnitudes, orientations
or mutual correlations. Under this uninformed prior assumption for the noise, we
consider all noise sources in the brain space to be uncorrelated, randomly oriented,
and possessing equal RMS amplitudes. Any deviations from this distribution reflect
subject's real brain activity and are therefore considered as the "sources of
interest." To apply this concept in practice, it is necessary to select a specific
RMS amplitude for the noise sources, which is accomplished in the following manner.

The measured covariance `R` is the sum of the covariance of the 'signal' (i.e.,
the brain activity of interest) and the noise (our uninformed prior): `R = S + N`,
where `S` is some positively defined matrix; `S > 0` . Therefore `R - N` must
be positively defined, which sets an upper limit on the RMS amplitude of the noise
sources. In this implementation, the software selects the RMS amplitude of the noise
sources to match this upper limit. Note that for the beamformer solutions
**specific setting of the noise amplitude** only affects the reconstructed source
pseudo-Z values but **does not affect the shape of the source wave forms**.

It is important to acknowledge that the covariance matrix `R` is inherently degenerate.
Although the number of EEG channels is 20, the rank of `R` typically falls within
the range of 12 to 18. Consequently, the above expressions involve pseudo-inverses
of `R`, which are also degenerate. However, when working with degenerate matrices
in Python using the numpy library, numerical issues may arise.

To mitigate these issues, the current software implementation replaces matrices `R`
and `R^-1` with the closest positively defined (non-degenerate) matrices.
This adjustment ensures numerical stability while introducing negligible effects
on the results, comparable to rounding errors in practice.

### Normalizing for group analyses
Regardless of the units chosen for the reconstructed source time courses, their
amplitudes can vary substantially among subjects. To prevent statistical biases
in group analyses, we take an additional step of **normalizing the weights** (and
consequently the reconstructed time courses) **by the sensor-level pseudo-Z**.

To achieve this normalization, we multiply each source weight `w` by a factor
`sqrt[trace(N)/trace(R)]`. By doing so, we ensure that all subjects' waveforms
are adjusted to have the same sensor level pseudo-Z, which is equal to 1. This
normalization helps to mitigate the amplitude differences among subjects and
promotes fair comparisons in group analyses.

### JSON configuration file
Configuration parameters for the source reconstruction step are defined in file
`src_reconstr_conf.json`. Note that JSON files can not contain comments;
therefore comments in the code below should be removed for this JSON to be used
in practice.

```python
{
	"bem_sol": "fsaverage-5120-5120-5120-bem-sol.fif",      # Precalculated BEM solution
	"source_space": "fsaverage-ico-3-src.fif",              # Precalculated source space
	"parcellation": "aparc",                                # Can be "aparc" or "aparc.a2009s" 
	"surface": "white",             # Surface to use for the source space
	"min_dist_to_skull_mm": 5.0,    # Min allowed distance between a source and the skull
	"max_condition_number": 1e10,   # Matrices with larger cond numbers considered degenerate
	"inverse_method": "beam",       # "beam" is THE ONLY choice for now
	"beam_type": "pz",              # Beamformer localizer type; no need to change
	"src_units": "pz",              # Either "source" or "pz" 
	"noise_upper_bound_tolerance": 1e-2,    # Accuracy of choosing max allowed noise amp
	"roi_time_course_method": "pca_flip"    # The way how a single ROI time course is obtained 
}
```

## Miscellaneous functions
Some utility scripts are located in the folder `.../eegfhabrainage/misc`. 

* `view_raw_eeg.py`: a simple EEG file viewer
* `view_inflated_brain_data.py`: function to plot user data over an inflated cortex
  surface and display cortex parcellation. In the same file -  
  `expand_data_to_rois()`: a helper function for expanding single data values referring
  to a set of *Regions Of Interest* (ROIs) to every surface vertex belonging to corresponding
  ROI.
* `plot_alpha_power.py`: an example of applying `view_inlfated_brain_data()` to plot
    distribution of alpha band spectral density over the cortex surface

Please refer to ["auto-generated documentation"](file://../doc/_build/html/index.html) for more details.

## Setting up Python virtual environment on Compute Canada (Alliance) cluster
The following steps should be performed to run the code on Compute Canada. The same setup can
be used on a local machine, except that commands `module load`, `deactivate` are not required, and
`--no-index`, `--no-download` flags should **not** be used.

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
        pip3 install mne-qt-browser      # If one wants to use QT backend
        pip3 install pyprep
        python3 -m pip install --no-index scikit-learn
        python3 -m pip install pyvistaqt # NOTE: only needed for interactive 3D graphics 
        python3 -m pip install nibabel

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
- From time to time you may need to install new modules in your virtual environment or update existing ones.
  For example to install `mymodule` and to upgrade MNE Python to its latest stable version, use:
```
	module load python/3.8.10
	module load scipy-stack
	cd <your working folder>
	source mne/bin/activate

	pip3 install --no-index --upgrade pip
	python3 -m pip install -U mne[hdf5] 
	python3 -m pip install mymodule

	deactivate
```
