# EEG preprocessing pipeline for FHA EDF dataset

## Summary

The code is aimed to preprocess clinical EEG recordings and make them a suitable input for later analyses and ML applications. 
Code uses the original EEG records in EDF format as input. The following steps (tasks) may be performed:

1. **Filtering, resampling and extracting of good data segments** of a target length. The output is the good segments in EDF format.
2. **Performing EEG PREP procedure and artifact removal**. The input is typically good segments obtained on the 1st step,
    and the output is the records in **.fif** format.
3. **Extracting hyperventilation and photic stimulation intervals from the original records**. The extracted
   intervals (segments) are saved in .EDF format.
4. **Performing beamformer reconstruction of source time courses from standard atlas locations ("Regions of Interest" - ROIs)**.
5. **Calculating power spectral densities (PSD)s of the sensor and source-reconstructed time courses.**
6. **Calculating Continuous Wavelet Transforms (CWTs) and estimating statistical parameters of the CWT amplitudes.**

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
[auto-generated documentation](../doc/_build/html/index.html).


***<span style="color:blue">NOTE</span>***: *<span style="color:blue">For the [documentation](../doc/_build/html/index.html) link to work one needs to:</span>*  
*<span style="color:blue">a) clone the repo to your local computer, and</span>*  
*<span style="color:blue">b) access this README file using a web browser that has the ["markdown file viewer extension"](https://chrome.google.com/webstore/detail/markdown-viewer/ckkdlimhmcjmikdlpkmbgfkaikojcbjk) installed.</span>*   
*<span style="color:blue">c) alternatively, on your local computer open a relative location `../doc/_build/html/index.html` directly in you browser</span>*  
*<span style="color:blue">Otherwise (i.e.on github) this link will just display a text view of the documentation's `index.html` file.</span>*

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
        N_ARRAY_JOBS = 100      # Number of parallel jobs to run on cluster
        ...
```
>> The **sbatch script** (see **`eeg_array_job.sbatch`** as an example):
```
        ...
        #SBATCH --array=0-99    # the last job index should be equal to N_ARRAY_JOBS - 1 
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

Detailed description of all arguments of each function or method are available in the [auto-generated documentation](../doc/_build/html/index.html).

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

        "discard_at_start_seconds": 420,        # time interval removed from the begining of the input record
        "target_frequency":         256,        # the sampling frequency of the output record
        "target_band":              [0.5, 55],  # the frequency band of the output record
        "target_segments":          1,          # Max number of good continuous segments to extract
        "target_length":            360,        # The good segment length in seconds
        "powerline_frq":            60.0,       # Power line main frequency, Hz
        "allow_upsampling":         false,      # Allow upsampling a record if its sampling rate
                                                # is smaller than requested

        # Parameters to identify flat intervals
        "flat_parms":
                {
                        "flat_max_ptp": 1e-06,  # max amplitude peak-to-peak value for the flat interval
                        "bad_percent": 50.0,    # min percentage of the time the channel's peak
                                                # to peak is below the 'flat_max_ptp' threshold
                                                # to be considered flat
                        "min_duration": 10.0    # minimum interval in seconds for all consecutive samples to
                                                # be below the 'flat_max_ptp' to indicate a flat interval
                },

        "HV_regexp":    "H.*V.*\\d+\\s*[MmIiNn]{3}",    # Regular expression to identify HV annotations like "HV 1 Min"
        "HV_end":       "END HV",                       # Hyperventilation end annotation, if present
        "hv_pad_interval": 30,                          # Padding interval in seconds around HV series. Final HV boundaries
                                                        # are set as follows:
                                                        # HV start = 1st HV mark - 60 - pad_interval
                                                        # HV end = last HV mark + 60 + pad_interval

        "photic_starts": ["Hz"],        # Keyword in annotation that marks the start of the photic stim
        "photic_ends": ["Off"],         # !! EXACT WORDING !! of the annotation that marks the end of the photic stim
        "photic_pad_interval": 30,      # Padding inteval in seconds around photic stim series

        "max_isi": 360,                 # Max interval in seconds between photic stimulations to consider those
                                        # belonging to the same photic stimulation series
        "max_rec_length": 3600          # Max allowed EDF file length in seconds (skip longer files)
}

```

## 2. EEG PREP procedure and artifact removal
### Running the code
This step is done by a top level script **`run_pyprep_ica.py`**. Most of related functions and classes are defined in
the source file `do_pyprep.py`. The main work horse is class `Pipeline`, which in turn uses class `PrepPipeline` imported from a `pyprep` library.
See more details in the ["auto-generated documentation"](../doc/_build/html/index.html).

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
        "montage": "standard_1020",     # As is
        "powerline_frq": 60.0,          # Powerline frq, Hz

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
        "plot_dpi": null,       # DPI for the plots saved as .PNG
                                # null will set DPI = screen resolution
        "plot_time_series":                     # Arguments to the call to raw.plot(); see
        {                                       # MNE Python docs for details   
                "duration": 20.0,               # Time interval to plot, s
                "start": 150.0,                 # Start of the plotted interval, s
                "color": null,
                "bad_color": "lightgray",
                "events": null,
                "event_color": "cyan",
                "scalings": {
                        "eeg": 50e-6,
                        "eog": 50e-6,
                        "ecg": 200e-6,
                        "misc": 500e-6,
                        "emg": 1e-3
                        },
                "remove_dc": true,
                "order": null,
                "show": true,
                "block": false,
                "clipping": 1.5,
                "show_first_samp": false,
                "group_by": "type",
                "butterfly": false,
                "event_id": null,
                "show_scrollbars": false,
                "show_scalebars": true,
                "time_format": "float"
        },

        "plot_psd":
        {
                "fmin": 0.0,                    # PSD frequency limits
                "fmax": 100.0,
                "ymin": 1e-1,                   # PSD values limits
                "ymax": 2e1,
                "fstep": 10.0,                  # Ticks step for X-axis
                "spect_log_x": true,            # Flags to use log scale
                "spect_log_y": true,
                "n_fft": 1024,                  # FFT size for welch() function
                "kwargs": {                     # Parameters for MNE Python
                        "average": false,       # Spectrum.plot() function
                        "dB": false,
                        "amplitude": true,
                        "xscale": "linear",
                        "ci": "sd",
                        "ci_alpha": 0.3,
                        "color": "black",
                        "alpha": null,
                        "spatial_colors": false,
                        "sphere": null,
                        "exclude": null,
                        "axes": null,
                        "show": false
                        }
        }
}

```

## 3. Extracting hyperventilation and photic stimulation intervals from the original records
This step is completely independent from steps 1, 2 and is only used to identify and store the
hyperventilation (HV) and photic stimulation (PS) segments, as the name suggests.

### Running the code
The top level scripts to execute are **`extract_hv_intervals.py`**, **`extract_ps_intervals.py`**
respectively. Internally both scripts call the same function `slice_edfs()` as in the
[first step](#1-filtering-resampling-and-extracting-of-good-data-segments).

When executing the code locally, use these commands in Linux terminal:
```
python extract_hv_intervals.py
python extract_ps_intervals.py
```
If running as an array job on the cedar cluster, use these commands in your sbatch script
(see `eeg_array_job.sbatch` for an example):
```
python extract_hv_intervals.py ${SLURM_ARRAY_TASK_ID}
python extract_ps_intervals.py ${SLURM_ARRAY_TASK_ID}
```

As usual, the main scripts **`extract_XX_intervals.py`** may need to be **modified** to set
the root input and output folders, etc. Please refer to section ["Running the code"](#running-the-code)
under the segmentation task
["1. Filtering, resampling and extracting of good data segments"](#1-filtering-resampling-and-extracting-of-good-data-segments),
for details. Note that these steps should only be applied to the original EDF records. 

### Performed operations
* First, the same **basic preprocessing operations as in the task 1 (segmentation) are done**, namely
    * The EDF file is loaded, except for the channels specified in the `exclude_channels` list.
    * Some of the channels may be renamed to ensure they are treated correctly by the MNE Python software
    * Based on the channel names, channel types are assigned.
    * All EEG, EOG and ECG channels are notch-filtered at power line frequencies
    * All EEG channels are additionally band-pass filtered to a `target_band`
    * All channels are resampled to a sampling frequency equal to the `target_frequency`

* **Hyperventilation intervals** are identified by markers (annotations) specified under keys
  *"HV_regexp"*, *"HV_end"* in the `preproc_conf.json` file. **Photic stimulation intervals** are
   identified by markers specified under keys *"photic_starts"*, *"photic_ends"* in the same file.
  No padding is applied to the extracted HV and PS segments.

* The resulting records are saved in EDF format.

### JSON configuration file
All HV and PS-related configuration parameters are stored in the JSON file `preproc_conf.json`.

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

## 5. Calculating power spectra
At this step power spectral densities (PSDs) for the preprocessed sensor space and source reconstructed data are calculated using the Welch method.
 
### Running the code
Corresponding top level script is **`run_welch.py`**; it also depends on source
file `do_src_reconstr.py`. 

When executing the code locally, run this command in Linux terminal from the `.../eegfhabrainage/preprocessing`
folder:
```
python run_welch.py
```

If running as an array job on the cedar cluster, use this command in your sbatch script
(see `eeg_array_job.sbatch` for an example):
```
python run_welch.py ${SLURM_ARRAY_TASK_ID}
```
In contrast to steps 1 to 4 described above, all input parameters for `run_welch.py` including input and output folders are defined in a dedicated JSON file `welch_input.json`. Therefore no modification of the Python source is required. Here is an example of such file:
```python
# ----------------------------------------------------------------------
# NOTE: this is a JSON file with comments. It requires using commentjson
# module to process, instead of a standard python json library
# ----------------------------------------------------------------------
# Input parameters for run_welch.py                  
# ----------------------------------------------------------------------
{
    "what": "sources",          # sensors' or 'sources' - which PSD to calculate
    "hospital": "Burnaby",          # Burnaby, Abbotsford, RCH, Surrey
    "source_scan_ids": null,    # null or a list of specific record IDs
    "view_plots": true,         # Flag to show interactive plots
    "plot_only": false,         # Flag to only plot already precalculated spectra",

    # IDs of records to plot if view_plots = true, or null, or []
    "plot_ids": ["2f8ab0f5-08c4-4677-96bc-6d4b48735da2",    # Burnaby
                "ffff1021-f5ba-49a9-a588-1c4778fb38d3",
                "81c0c60a-8fcc-4aae-beed-87931e582c45",
                "81be60fc-ed17-4f91-a265-c8a9f1770517",
                "57ea2fa1-66f1-43f9-aa17-981909e3dc96",
                "fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2"
            ],

    # Channels/ROIs to plot if view_plots = true
    "plot_chnames": {
            "sensors": ["O1","O2","P3","P4","Pz"],
            "sources": ["G_occipital_middle-lh", "G_occipital_middle-rh", "G_occipital_sup-lh",
                "G_occipital_sup-rh", "Pole_occipital-lh", "Pole_occipital-rh"]
            },

    "N_ARRAY_JOBS": 100,        # Num of parallel jobs to run on cluster; should match
                                # total number of jobs specified in .sbatch file
    "verbose": "INFO",          # Can be ‘DEBUG’, ‘INFO", ERROR", "CRITICAL", or "WARNING" (default)

    # Input and output root folders depending on the host computer
    "hosts": {
        "ub20-04": {
            "cluster_job": false,   # this is not an array job on the CC cluster
            "sensors": {
                "data_root": "/data/eegfhabrainage/after-prep-ica", # input time courses here
                "out_root": "/data/eegfhabrainage/welch",           # output PSDs here
            },
            "sources": {
                "data_root": "/data/eegfhabrainage/src-reconstr",
                "out_root": "/data/eegfhabrainage/src-welch"
            }
        },
        "cedar": {
            "cluster_job": true,

            "sensors": {
                "data_root": "/project/6019337/databases/eeg_fha/preprocessed/001_a01_01",
                "out_root": "/home/amoiseev/projects/rpp-doesburg/amoiseev/data/eegfhabrainage/welch"
            },
            "sources": {
                "data_root": "/project/6019337/databases/eeg_fha/beamformed/001_a01_01/destrieux",
                "out_root": "/home/amoiseev/projects/rpp-doesburg/amoiseev/data/eegfhabrainage/src-welch"
            }
        },
        "other": {
            "cluster_job": false,   # this is not an array job on the CC cluster
            "sensors": {
                "data_root": "after-prep-ica", # input time courses here
                "out_root": "welch",           # output PSDs here
            },
            "sources": {
                "data_root": "src-reconstr",
                "out_root": "src-welch"
            }
        }
    }   # end of "hosts"
}

```
The meaning of the input parameters is explained in comments.

The general configuration parameters for this step which are not likely to be frequently changed are given in a JSON file `welch_conf.json`, which looks like this:
```python
{
    "freq_range": [1.0, 55.0],	# Frequency range (inclusive), Hz
    "freq_points": 109,		# Number of frequency bins for the PSD

    # Extra parameters (in addition to data, fs and nperseg) passed to the
    # scipy.signal.welch() function 
    "welch":
    {
        "window": "hann",
        "noverlap": null,
        "nfft": null,
        "detrend": "constant", 
        "return_onesided": true,
        "scaling": "density",
        "axis": -1,
        "average": "mean"
    }
}
```
For the sensor space spectra, the input records are expected in `.fif` format; typically those obtained after the PREP/ICA step will be used. For the source space spectra, the records in `.hdf5` format obtained at the source reconstruction step should be used. The results are also stored in `.hdf5` format and contain: a list of the channel names, a list of frequencies in Hz, and calculated PSDs for each channel measured in "units"^2/Hz.  "Units" are Volts for sensor channels PSDs. For the source level PSDs the units depend on the setting of `src_units` parameter in `src_reconstr_conf.json` file at the time of running the source reconstruction step. That will be A*m when `src_units` was set to 'source', or amplitude pseudo-Z when `src_units` was set to 'pz'. 

### Performed operations
The power spectral density estimates for each channel in the input records are calculated using the Welch method.

## 6. Calculating Continuous Wavelet Transforms (CWTs)
This step may be performed for both the sensor and the source space data. For each channel of the input record, a CWT is calculated using a [complex Morlet wavelet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet2.html) as a base function. Then channels CWTs are used to estimate the following parameters of statistical distributions of the CWT amplitudes: *mean, median, standard deviation, skew*, and *kurtosis*. Additionally an exponentiated Weibull distribution is fit to the CWT amplitude data to obtain 4 parameters: `a`, `c`, `loc`, `scale` (see [the scipy documentation](http://scipy.github.io/devdocs/reference/generated/scipy.stats.exponweib.html) for reference). These results are stored as an .hdf5 file.
 
### Running the code
Corresponding top level script is **`run_cwt.py`**.

When executing the code locally, run this command in Linux terminal from the `.../eegfhabrainage/preprocessing`
folder:
```
python run_cwt.py
```

When running as an array job on the cedar cluster, use this command in your sbatch script
(see `eeg_array_job.sbatch` for an example):
```
python run_cwt.py ${SLURM_ARRAY_TASK_ID}
```
All input parameters for `run_cwt.py` including input and output folders are defined in a dedicated JSON file `cwt_input.json`. Thus no modification of the Python source is required. Here is an example of such file:
```python
# ----------------------------------------------------------------------
# NOTE: this is a JSON file with comments. It requires using commentjson
# module to process, instead of a standard python json library
# ----------------------------------------------------------------------
# Input parameters for run_cwt.py                  
# ----------------------------------------------------------------------
{
    "what": "sources",          # sensors' or 'sources' - which CWT to calculate
    "hospital": "Abbotsford",          # Burnaby, Abbotsford, RCH, Surrey
    "source_scan_ids": null,    # null or a list of specific record IDs
    "view_plots": false,         # Flag to show interactive plots
    "plot_only": false,         # Flag to only plot already precalculated spectra",

    # IDs of channels to plot if view_plots = true, or null, or []
    "plot_ids": ["57ea2fa1-66f1-43f9-aa17-981909e3dc96",
                 "2f8ab0f5-08c4-4677-96bc-6d4b48735da2"],    # Burnaby

    # Channels/ROIs to plot in distribution plots if view_plots = true
    "plot_chnames": {
            "sensors": ["O1","O2","P3","P4","Pz"],
            "sources": ["G_occipital_middle-lh", "G_occipital_middle-rh", "G_occipital_sup-lh",
                "G_occipital_sup-rh", "Pole_occipital-lh", "Pole_occipital-rh"]
            },

    # X, Y limits for distribution plots, or null
    "plot_xlim": {
            "sensors": null,
            "sources": null 
            },

    "plot_ylim": {
            "sensors": null,
            "sources": null 
            },

    "plot_nx": 200,     # Number of x-points for the amplitude distribution plots

    # Frequencies to plot in distribution plots if view_plots = true
    "plot_freqs": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 43.0, 55.0],

    # Single scan ID to print out the results, or null
    "print_id": "99a27a66-4e90-41d9-8139-2094b562b277",
    "print_chan": "G_occipital_sup-lh",  # Channel name for printing single channel results, or null
    "print_freq": 10.0,         # Frequency value for printing single frequency results, or null

    "N_ARRAY_JOBS": 100,        # Num of parallel jobs to run on cluster; should match
                                # corresponding settings in .sbatch file
    "verbose": "INFO",          # Can be ‘DEBUG’, ‘INFO", ERROR", "CRITICAL", or "WARNING" (default)

    # Input and output root folders depending on the host computer
    "hosts": {
        "ub20-04": {
            "cluster_job": false,   # this is not an array job on the CC cluster
            "sensors": {
                "data_root": "/data/eegfhabrainage/after-prep-ica", # input time courses here
                "out_root": "/data/eegfhabrainage/cwt",             # output CWT results here
            },
            "sources": {
                "data_root": "/data/eegfhabrainage/src-reconstr",
                "out_root": "/data/eegfhabrainage/src-cwt"
            }
        },
        "cedar": {
            "cluster_job": true,

            "sensors": {
                "data_root": "/project/6019337/databases/eeg_fha/preprocessed/001_a01_01",
                "out_root": "/home/amoiseev/projects/rpp-doesburg/amoiseev/data/eegfhabrainage/cwt"
            },
            "sources": {
                "data_root": "/project/6019337/databases/eeg_fha/beamformed/001_a01_01/destrieux",
                "out_root": "/home/amoiseev/projects/rpp-doesburg/amoiseev/data/eegfhabrainage/src-cwt"
            }
        },
        "other": {
            "cluster_job": false,   # this is not an array job on the CC cluster
            "sensors": {
                "data_root": "after-prep-ica", # input time courses here
                "out_root": "cwt",           # output PSDs here
            },
            "sources": {
                "data_root": "src-reconstr",
                "out_root": "src-cwt"
            }
        }
    }   # end of "hosts"
}

```
The meaning of the input parameters should be clear from the comments.

The general configuration parameters for this step which are not likely to be modified frequently are given in a JSON file `cwt_conf.json`, which looks like this:
```python
# ----------------------------------------------------------------------
# NOTE: this is a JSON file with comments. It requires using commentjson
# module to process, instead of a standard python json library
# ----------------------------------------------------------------------
# Configuration file for run_cwt.py
{
    "freq_range": [1.0, 55.0],
    "freq_points": 50,      # Number of log-spaced frequencies, including the end points

    "fit_distribution": "exponweib",

    # Parameters of spectral amplitude distribution to calculate for each frequency
    # and keyword arguments to pass to corresponding methods
    "spect_parms":
    {
        "mean": {"axis": 1},
        "median": {"axis": 1},
        "std": {"axis": 1, "ddof": 1},
        "skew": {"axis": 1, "bias": false},
        "kurtosis": {"axis": 1, "fisher": true, "bias": false, "nan_policy": "raise"},

        # The distribution and fit parameters. The kwarg passed should be 
        # "parm": <parameter-name-tself>
        "ew_a": {"parm": "ew_a"}, "ew_c": {"parm": "ew_c"}, "ew_loc": {"parm": "ew_loc"},
        "ew_scale": {"parm": "ew_scale"},
        "ew_fit_stat": {"parm": "ew_fit_stat"}, "ew_fit_pval": {"parm": "ew_fit_pval"}
    },

    "zscore":   # Additional optional arguments for the call scipy.stats.zscore(data, axis = 1, **kwargs])
    {           # See scipy.stats.zscore() docs for the meaning of those
        "ddof": 0,
        "nan_policy": "propagate"
    },

    "welch":
    {
        "window": "hann",
        "noverlap": null,
        "nfft": null,
        "detrend": "constant", 
        "return_onesided": true,
        "scaling": "density",
        "axis": -1,
        "average": "mean"
    }
}
```
The input data for this step are the same as for the PSD step described above. Namely for the sensor space data, the input records are expected in `.fif` format; typically those obtained after the PREP/ICA step will be used. For the source space data, the records in `.hdf5` format obtained at the source reconstruction step should be used.

The processing results of each record are accumulated in an [xarray.DataArray](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html#xarray.DataArray) object. This object is then stored as `.hdf5` file using a dedicated xarray function. The stored data contains a list of the channel names, a list of frequencies in Hz, and the estimated CWT amplitude distribution parameters described earlier.

### Performed operations
The continuous wavelet transforms of ***standardized*** (that is *z-scored*) input data are calculated for each channel in a form of complex valued time courses for each frequency line. The frequencies are selected so that ***their decimal logarithms are linearly spaced*** in the requested frequency range. Then absolute values of the CWT coefficients are found to obtain CWT amplitude distributions. For these amplitude distributions a) the means, medians, standard deviations and skews are estimated b) an exponentiated Weibull distributions are fit to the obtained data. Finally, the CWTs themselves are discarded and only estimated statistical parameters of the amplitude distributsions are preserved and stored in `.hdf5` file. For details regarding how it is implemented, please refer to the description of `TFData` class in the [documentation](../doc/_build/html/index.html).

Note that the Weibull distribution fit step is extremely time consuming therefore provisions are made to parallelize calculations for different frequency lines. One can take advantage of this by specifying an appropriate `--cpus-per-task` parameter in the sbatch job when running on the cluster.

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
* `welch_logf.py`: calculate power spectrum by Welch method for logarithmically
    spaced frequencies
* `utils.py`: a set of small utility functions; see [documentation](../doc/_build/html/index.html) for details

Please refer to [auto-generated documentation](../doc/_build/html/index.html) for more details.

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
        module load scipy-stack/2022a

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

        python3 -m pip install commentjson      # For processing JSON files with comments
        python3 -m pip install xarray           # xarray used to save CWT results
        python3 -m pip install h5netcdf         # For saving xarrays to hdf5

        # Optional
        python3 -m pip install PyWavelets       # wavelet transforms
        python3 -m pip install pactools         # Package for phase-amp coupling

        # Only needed to generate the documentation using sphinx.
        # This installs the read-the-docs theme:
        python3 -m pip install myst_parser
        python3 -m pip install sphinx_rtd_theme

        deactivate
```
- In your sbatch scripts, use commands
```
        module load python/3.8.10
        module load scipy-stack/2022a
        cd <your working folder>
        source mne/bin/activate

        < run your python program >
        
        deactivate
```
- From time to time you may need to install new modules in your virtual environment or update existing ones.
  For example to install `mymodule` and to upgrade MNE Python to its latest stable version, use:
```
        module load python/3.8.10
        module load scipy-stack/2022a
        cd <your working folder>
        source mne/bin/activate

        pip3 install --no-index --upgrade pip
        python3 -m pip install -U mne[hdf5] 
        python3 -m pip install mymodule

        deactivate
```
