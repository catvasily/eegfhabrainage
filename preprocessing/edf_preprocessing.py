import mne
import os
import os.path as op
import glob
from individual_func import write_mne_edf
from mne.preprocessing import annotate_amplitude
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import zscore
import json

JSON_CONFIG_FILE = "preproc_conf.json"
'''Default name (without a path) for the JSON file with preprocessing parameters. It is expected
to reside in the same folder as the *edf_preprocessing.py* source file.

'''

_JSON_CONFIG_PATHNAME = os.path.dirname(__file__) + "/" + JSON_CONFIG_FILE
'''Automatically generated fully qualified pathname to the default JSON config file

'''

def read_edf(filepath, *, conf_json = _JSON_CONFIG_PATHNAME, conf_dict = None, target_channels = None,
	exclude_channels = None):
    '''
    Reads an EDF file with the MNE package, creates the Raw EDF object. 
    Excludes some channels to keep only target ones.
    Prints warning in case the file doesn't have all 
    neeeded channels, doesn't return object in this case.
    
    Args:
      filepath (str): EDF file pathname
      conf_json (str): pathname of a json file with configuration parameters; in
         particular it must contain keys *"target_channels"* and *"exclude_channels"*.
         Default configuration file name is given by :data:`JSON_CONFIG_FILE` constant
      conf_dict (dict): a dictionary with configurartion parameters; in
         particular it must contain keys *"target_channels"* and *"exclude_channels"*.
         If both *conf_json* and *conf_dict* are given, the latter is used.
      target_channels (list of str): a list of mandatory channels to read from the EDF
         file. If supplied, *target_channels* list will be used instead of the one provided
         by either of the *conf_...* arguments.
      exclude_channels (list of str): a list of channels to be excluded when reading
         the EDF file. If supplied, *exlclude_channels* list will be used instead of the one provided
         by either of *conf_...* arguments.
    Returns:
      Raw (mne.Raw): raw EDF object

    Note:
        At least one of the arguments: *conf_json* or *conf_dict* must be not None. If both *conf_json*
        and *conf_dict* are omitted in the function call, the default settings will be read from 
       :data:`JSON_CONFIG_FILE` which should be present in the same folder as the *edf_preprocessing.py*.
       
    '''
    # Validate args
    if (conf_json is None) and (conf_dict is None):
        raise ValueError("At least one of the arguments 'conf_json' or 'conf_dict' should be specified")

    if conf_dict is None:
        # Read configuraion from a json file
        with open(conf_json, "r") as fp:
            conf_dict = json.loads(fp.read())

    # Set channel lists from the dictionary when not specified
    if target_channels is None:
        target_channels = conf_dict["target_channels"]

    if exclude_channels is None:
        exclude_channels = conf_dict["exclude_channels"]

    # At this point both target_channels and exclude_channels are set

    # read EDF file while excluding some channels
    data = mne.io.read_raw_edf(filepath, exclude = exclude_channels, verbose='warning', preload=True)
    
    # the list of target channels to keep
    target_channels = set(target_channels)
    current_channels = set(data.ch_names)
    
    # checking whether we have all needed channels
    if target_channels == current_channels:
        return data
    else:
        print(filepath, "File doesn't have all needed channels")

class PreProcessing:
    """The class's aim is preprocessing clinical EEG recordings (in EDF format)
    and make them a suitable input for a later analysis and ML applications.
     
    The class instantiates a preprocessing object which 
    carries a Raw EDF file through a sequence of operations: 
      
    #. resample each recording's signal to traget frequency
    #. keeps only signal with frequencies in selected range
    #. identifies timestamps of hyperventilation (HV), photic stimulation (PhS)
       and flat (zero) signal (together - "bad" intervals)
    #. extract EEG segment(s) of needed length from "good" intervals
         
    Then the object can save extracted segment(s) into new EDF files 
    OR return a Pandas DataFrame with data.
        
    Args:
        filepath (str): the EDF file pathname
        conf_json (str): pathname of a json file with configuration parameters.
           The default configuration file name is given by :data:`JSON_CONFIG_FILE` constant.
        conf_dict (dict): a dictionary with configurartion parameters.
           If both *conf_json* and *conf_dict* are given, the latter is used.
        target_channels (list of str): a list of mandatory channels to read from the EDF
           file. 
        exclude_channels (list of str): a list of channels to be excluded when reading
           the EDF file. 
        target_frequency (int): the final EEG frequency after resampling
        lfreq (float): lower frequency boundary of the signal to keep
        hfreq (float): higher frequency boundary of the signal to keep
        flat_parms (dict): parameters for flat intervals selection. Should contain the
            following keys:
            *'flat_max_ptp'* - the channel's amplitude max peak-to-peak value (in channel's
            units) for this channel to be marked as flat;
            *'bad_percent'* - min percentage of the time the channel's peak
            to peak is below the *'flat_max_ptp'* threshold to be considered flat;
            *'min_duration'* - minimum interval in seconds for all consecutive samples to
            be below the *'flat_max_ptp'* to indicate a flat interval.

    Note:
        * At least one of the arguments: *conf_json* or *conf_dict* must be not None. If both *conf_json* and *conf_dict*
          are omitted in the constructor, the default settings will be read from :data:`JSON_CONFIG_FILE` which should
          be present in the same folder as the *edf_preprocessing.py*.
        * When not None, the explicitly specified parameter values will be used instead of the values
          given by corresponding keys in *conf_json* or *conf_dict*.

    **Attributes**

    Attributes:
        filename (str): EDF file path
        conf_dict (dict): a dictionary with configuration parameters
        target_channels (list of str): a list of mandatory channels to read from the EDF
           file. 
        exclude_channels (list of str): a list of channels to be excluded when reading
           the EDF file.
        target_frequency (int): the final EEG frequency after resampling
        raw (mne.Raw): the raw EDF object
        sfreq (float): initial sampling frequency of EEG
        flat_parms (dict): parameters for flat intervals selection (see above)
        bad_intervals (list): a list of the [start, end] pairs with timestamps in seconds,
            indicating starts and ends of bad intervals (HV, PhS, flat signal)
        clean_intervals (list): a list of the [start, end] pairs with timestamps in seconds,
            indicating starts and ends of segments to be extracted

    **Methods**
            
    """
    # This part of the class description was moved out of a docstring to avoid sphinx warnings
    # about duplicate descriptions. Now only descriptions from the methods themselves will
    # be present in the generated documentation.
    # Methods:
    #     flat_intervals: returns list of [start, end] timestamps in seconds of zero signal
    #     hyperventilation: returns list of [start, end] timestamps in seconds of HV
    #     photic_stimulation: returns list of [start, end] timestamps in seconds of PhS
    #     extract_good: calling this method defines clean_intervals; 
    #         it doens't manipulate data itsef, just returns the intervals' timestamps
    #     create_intervals_data: returns DataFrame with the EEG data based on clean_intervals
    #     save_edf: write new EDF files based on clean_intervals timestamps

    def __init__(self, filepath, *, conf_json = _JSON_CONFIG_PATHNAME, conf_dict = None,
                 target_channels = None, exclude_channels = None,
                 target_frequency = None, lfreq = None, hfreq = None, flat_parms = None):
        """Constructor args are explained in the class description above.
            
        """
        # Validate args
        if (conf_json is None) and (conf_dict is None):
            raise ValueError("At least one of the arguments 'conf_json' or 'conf_dict' should be specified")

        if conf_dict is None:
            # Read configuraion from a json file
            with open(conf_json, "r") as fp:
                conf_dict = json.loads(fp.read())

        # Set the missing arguments from the dictionary
        if target_channels is None:
            target_channels = conf_dict["target_channels"]

        if exclude_channels is None:
            exclude_channels = conf_dict["exclude_channels"]

        if target_frequency is None:
            target_frequency = conf_dict["target_frequency"]

        if lfreq is None:
            lfreq = conf_dict["target_band"][0]

        if hfreq is None:
            hfreq = conf_dict["target_band"][1]

        if flat_parms is None:
            flat_parms = conf_dict["flat_parms"]

        self.filename = filepath
        self.conf_dict = conf_dict
        self.target_channels = target_channels
        self.exclude_channels = exclude_channels
        self.target_frequency = target_frequency
        self.raw = read_edf(filepath, target_channels = target_channels, exclude_channels = exclude_channels)
        self.raw.filter(l_freq=lfreq, h_freq=hfreq)
        self.sfreq = dict(self.raw.info)['sfreq']
        self.flat_parms = flat_parms

        if(self.sfreq != self.target_frequency):
            self.raw.resample(self.target_frequency)
        
        self.clean_intervals = []
        self.intervals_df = pd.DataFrame()
        mne.set_log_level('warning')
        
    def flat_intervals(self):
        '''Identify beginning and end times of flat signal
        
        Returns:
            intervals (list): List of *[start, end]* time values
                for each interval
            
        '''
        annot_bad_seg, flat_chan = annotate_amplitude(self.raw, 
                                       bad_percent = self.flat_parms["bad_percent"],
                                       min_duration = self.flat_parms["min_duration"],
                                       flat = self.flat_parms["flat_max_ptp"],
                                       picks=None, verbose=None)
        intervals = []

        for i in annot_bad_seg:
            start = list(i.items())[0][1]
            duration = list(i.items())[1][1]
            end = start+duration
            intervals.append([start,end])
        return intervals


    def hyperventilation(self):
        """Identify beginning and end of hyperventilation from EEG data

        Returns:
            intervals (list): [[start, end]] or []: a list of start and end times of hyperventilation
                intervals. **Currently contains either a single interval**, or is empty.
             
        """

        start = np.nan
        end = np.nan
        parms = self.conf_dict["hyperventilation"]

        for position, item in enumerate(self.raw.annotations.description):
            if item in parms["hv_1min_start_notes"]:
                start = self.raw.annotations.onset[position] - parms["hv_pad_interval"]

            if item in parms["hv_1min_end_notes"]:
                end = self.raw.annotations.onset[position] + (parms["hv_pad_interval"] - int(item.split(' ')[2]))

        if np.isnan(start):
            for position, item in enumerate(self.raw.annotations.description):
                if item in parms["hv_start_notes"]:
                    # AM: was:
                    # 	start = self.raw.annotations.onset[position] - 30
                    # This prepends with 30 seconds, while in all other cases pre/post-
                    # pending is done with parms["hv_pad_interval"] seconds. Corrected.
                    start = self.raw.annotations.onset[position] - parms["hv_pad_interval"]

        if np.isnan(end):
            for position, item in enumerate(self.raw.annotations.description):
                if item in parms["hv_end_notes"]:
                    end = self.raw.annotations.onset[position] + parms["hv_pad_interval"]

        # when hyperventilation is present
        # eliminate the corresponding segment
	# AM: was:
        #	if start != np.nan and end != np.nan:
	# This is wrong, because np.nan != np.nan returns True. Thus when both
	# start and end are NaNs the returned interval is [NaN, NaN] instead of
	# empty []. The correct test is 
        if (not np.isnan(start)) and (not np.isnan(end)):
            return [[start, end]]
        else:
            # AM: ?? This will return [] if any of (start, end) is NaN. This is
            # a correct behavior only provided both starting and ending annotations are 
            # ALWAYS present. Is it indeed the case? Not sure - so added a check here:
            assert np.isnan(start) and np.isnan(end), \
                'Only one end of a hyperventilation interval is found'	# Assert when only one of them is NaN
            return []

    def photic_stimulation(self):
        """Identify beginning and end times of photic stimulation.
           
        Returns:
            intervals (list): [[start, end]] or []: a list of start and end times of hyperventilation
                intervals. **Currently contains either a single interval**, or is empty.
             
        """
        
        # store times when stimulation occurs
        stimulation = []
        
        # loop over descriptions and identify those that contain frequencies
        for position, annot in enumerate(self.raw.annotations.description):
            for kword in self.conf_dict["photic_stim_keywords"]:
                if kword in annot:
                    # record the positions of stimulations
                    stimulation.append(position)
                    break
        
        # provided stimulation has occured
        if len(stimulation)>1:
            # identify beginning and end
            start = self.raw.annotations.onset[stimulation[0]]
            end = self.raw.annotations.onset[stimulation[-1]] + self.raw.annotations.duration[stimulation[-1]]
            return [[start, end]]    
        else:
            return []
       
        # AM: ?? this code is unreachable - commented out
        # null value when no stimulation is present
        # return None

    def extract_good(self, target_length, target_segments):
        """This function calls the functions above to identify "bad" intervals and
        updates the attribute :data:`clean_intervals` with timesptamps to extract
        
        Args:
            target_length (float): length in seconds of the segments to be extracted
                from this EEG recording
            target_segments (int): a total number of the segments to extract 
                from this EEG recording

        Returns:
            None
             
        """
        
        self.bad_intervals = []
        # calling functions to identify different kinds of "bad" intervals
        self.bad_intervals.extend(self.flat_intervals())
        self.bad_intervals.extend(self.hyperventilation())
        self.bad_intervals.extend(self.photic_stimulation())
        self.bad_intervals.sort()	# This sorts intervals in place so that they start in ascending order
					# If the starts match, then the ends are in ascending order
        
        self.clean_part = self.raw.copy()
        tmax = len(self.raw)/self.target_frequency
                
        # Add 'empty' bad intervals in the beginning and in the end for furhter consistency
        skip_length = self.conf_dict["discard_at_start_seconds"]
        self.bad_intervals.insert(0,[0, skip_length]) # <--- SET FIRST "discard_at_start_seconds" interval AS BAD BY DEFAULT
        self.bad_intervals.append([tmax, tmax])
        # Construct temporary dataframe to find clean interval in EDF
        tmp_df = pd.DataFrame(self.bad_intervals, columns=['start', 'end'])
        
        # Define end of the clean interval as a start of next bad interval
        tmp_df['next_start'] = tmp_df['start'].shift(periods=-1)	# Shift series 'start' one step back
	# As there always is a bad interval at the start, the first value in the 'next_start' is the end
	# of the 1st *good* interval, and so on. Note that this operation adds NaN at the end of the column

        tmp_df.iloc[-1,-1] = tmax # <= Assign end of edf file as the end of last clean interval
				  # This replaces the trailing NaN in the 'next_start'
        
        # Handle cases when bad intervals overlap
        prev_value = 0
        new_ends = []
        for value in tmp_df['end']:
            if prev_value > value :
                new_ends.append(prev_value)
            else:
                new_ends.append(value)
                prev_value = value
        tmp_df['cumulative_end'] = new_ends
	# Now the bad intervals defined as [start, cumulative_end] may have different starts
	# but always the same ends
        
        # Calculate lengths of clean intervals
        tmp_df['clean_periods'] = tmp_df['next_start'] - tmp_df['cumulative_end']
	# Note that when there are overlapping intervals there will be negative values
	# in the 'clean_periods' column
        
        # Check whether there is at least 1 clean interval with needed target length
        if tmp_df[tmp_df['clean_periods'] >= target_length].shape[0] == 0:
            self.resolution = False
            pass	# AM: ?? does not seem that one needs 'pass' here
        else:    
            # if there is at least one clean segment of needed length, it updates clean_intervals list
            self.resolution = True
            
            # check how many availabe segments of needed length the whole recording has
            total_available_segments = (tmp_df[tmp_df['clean_periods'] > 0]['clean_periods'] // target_length).sum()
            
            # if we need 5 segments, and the recording has more, it extracts 5; 
            # if the recording has less than 5, let's say only 3 segments, it extracts 3
            if target_segments < total_available_segments:
                n_samples = target_segments
            else:
                n_samples = total_available_segments
                
            starts = list(tmp_df[tmp_df['clean_periods'] > 0]['cumulative_end'])
            n_available_segments = list(tmp_df[tmp_df['clean_periods'] > 0]['clean_periods'] // target_length)
            
            # updates clean_intervals attribute with timestamps
            # starting from the first available intervals
            for i in range(len(n_available_segments)):
                current_start = int(starts[i])
                for s in range(int(n_available_segments[i])):
                    self.clean_intervals.append(
                    (
                        int(current_start), 
                        int(current_start + target_length)
                    )
                    )
                    #print(s, self.clean_intervals)
                    current_start += target_length
                    if len(self.clean_intervals) >= n_samples:
                        break
                if len(self.clean_intervals) >= n_samples:
                    break

    def create_intervals_data(self):
        """ The function updates and returns intervals_df - a DataFrame 
        with the EEG data based on timestamps from :data:`clean_intervals`.
        Prints warning if no clean segments were found.

        The DataFrame has the following columns:

        - scan_id - ID of the EEG recording.
        - interval_start - timestamp in datapoints of the segment start.
        - interval_length - length in datapoints of the segment.
        - data - numpy array of the EEG amplitude data, with shape
          (<n_target_channels>, length in seconds x sampling frequency).
            
        Returns:
            self.interval_df (DataFrame): a dataframe with extracted segments
             
        Note:
            Saving this dataframe into a CSV file will truncate the content
            of the 'data' column and convert it to a string, so the data will be lost.
            Recommend to save the data into .npy file separately and keep the .csv
            for later matching with labels.
           
        """
        
        # check if there are available clean segments
        if self.resolution:
            # extracting scan ID from EDF file path, and prepare DF structure
            ids = np.repeat(self.filename.split('/')[-1].split('.')[0], len(self.clean_intervals))
            intervals_data = []
            interval_starts = []
            interval_lengths = []
            
            for i in range(len(self.clean_intervals)):
                interval_start = self.clean_intervals[i][0] * self.target_frequency
                interval_end = self.clean_intervals[i][1] * self.target_frequency
                
                interval_data = self.clean_part.get_data(start=interval_start, stop=interval_end)
                # apply zscore normalization to the amplitude values
                interval_data = zscore(interval_data, axis=1)
                
                intervals_data.append(interval_data)
                interval_starts.append(interval_start)
                interval_lengths.append(interval_end - interval_start)
                
            self.intervals_df['scan_id'] = ids
            self.intervals_df['interval_start'] = interval_starts
            self.intervals_df['interval_length'] = interval_lengths
            self.intervals_df['data'] = intervals_data
            
            return self.intervals_df
        else:
            print('Found no clean intervals of the specified length')
            
    def save_edf(self, folder, filename):
        """ The function writes out new EDF file(s) based on :data:`clean_intervals` timestamps.
        It saves each segment into a separate EDF file, with suffixes "[scan_id]_1",
        "[scan_id]_2", etc. 
        
        Args:
            folder (str):    where to save the new EDF files
            filename (str) : main name for the output files (suffix will be added for more than 1 files)

        Returns:
            None

        """
        
        # check if there are available clean segments
        if self.resolution:
            for n in range(len(self.clean_intervals)):
                interval_start = self.clean_intervals[n][0]
                interval_end = self.clean_intervals[n][1]
                
                tmp_raw_edf = self.clean_part.copy()
                
                tmp_raw_edf.crop(interval_start, interval_end, include_tmax=False)
                
                if n > 0:
                    scan_id = filename.split('.')[0]
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+scan_id + '_' + str(n)+'.edf', overwrite=True)
                else:
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+filename, overwrite=True)
        else:
            print('Found no clean intervals of the specified length')
            
            
def slice_edfs(source_folder, target_folder, target_length, *, conf_json = _JSON_CONFIG_PATHNAME, conf_dict = None,
               source_scan_ids = None, target_frequency = None, lfreq = None, hfreq = None, target_segments=1, nfiles=None):
    """ The function runs a pipeline for preprocessing and extracting 
    clean segment(s) of requested length from multiple EDF files.
    It takes in a list of EDF file names and preprocessing parameters, 
    looks up for the files in source folder, and performs preprocessing 
    and extraction.
    
    Args:
        source_folder (str): a pathname to the folder with EDF files 
        target_folder (str): a pathname to the output folder where the extracted segments
            will be saved in EDF format
        conf_json (str): pathname of a json file with configuration parameters.
           The default configuration file name is given by :data:`JSON_CONFIG_FILE` constant.
        conf_dict (dict): a dictionary with configurartion parameters.
           If both *conf_json* and *conf_dict* are given, the latter is used.
        target_length (float): the length of each of the extracted segments in seconds
        source_scan_ids (list of str or None) : a list of short EDF file names without .edf
            extention to preprocess. If None, all .edf files in the source directory will
            be preprocessed, up to a limit set by the *nfiles* argument
        target_frequency (num or None): the final EEG frequency after resampling; if not specified
            the default for the :class: `PreProcessing` will be used 
        lfreq (float or None): the lower frequency boundary of the EEG signal in Hz; if not specified
            the default for the :class: `PreProcessing` will be used
        hfreq (float or None): the higher frequency boundary of the EEG signal in Hz; if not specified
            the default for the :class: `PreProcessing` will be used
        target_segments (int): the maximum number of segments to extract from each EDF file;
            default=1
        nfiles (int or None): the max number of the source files to preprocess; (default = None = no limit)
          
    Returns:
        None

    """
    existing_edf_names = os.listdir(source_folder)

    if source_scan_ids is None:
        scan_files = existing_edf_names
    else:
        scan_files = [scan_id + '.edf' for scan_id in source_scan_ids]

    i = 0
    
    for f in scan_files:
        if f in existing_edf_names:
            path = source_folder + '/' + f

            try:
                # Initiate the preprocessing object, resample and filter the data
                p = PreProcessing(path, conf_json = conf_json, conf_dict = conf_dict,
                                  target_frequency=target_frequency, lfreq=lfreq, hfreq=hfreq)

                # This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
                p.extract_good(target_length=target_length, target_segments=target_segments)

                # Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
                p.save_edf(folder=target_folder, filename = f)
            
                i += 1
            except:
                print('Extraction failed for file ' + f)
            
            if i % 100 == 0 and i != 0:
                print(i, 'EDFs processed')
                
            if i == nfiles:
                break

    print('\nslice_edfs(): total of {} input EDF records processed.'.format(i))
            
def load_edf_data(source_folder, labels_csv_path):
    """The function loads multiple EDF files and returns data 
    with lables suitable for analysis and machine learning models.
    
    Args:
        source_folder (str): folder with EDF files
        labels_csv_path (str): CSV dile containing scan_ids and label (age)
        
    Returns:
        X (NumPy array): EEG amplitudes from EDF files, having
           shape of ([n_samples], 20, [length in seconds] x [sampling frequency])
        labels (NumPy array): age labels corresponding to each sample from X
        
    Note:
        This function fits a very specific CSV file 'age_ScanID.csv', 
        so it's looking for columns 'ScanID' for scan ids, 'AgeYears' for age labels. 
        You might want to adjust column names in your own file or change this function.
         
    """
    
    files = os.listdir(source_folder)

    df = pd.DataFrame()

    for file in files:
        if file.endswith('.edf'):
            rawedf = read_edf(source_folder + '/' + file)
            data = rawedf.get_data()

            tmp_df = pd.DataFrame()

            tmp_df['scan_id'] = [file.split('.')[0].split('_')[0]]
            tmp_df['data'] = [data]

            df = pd.concat([df, tmp_df], axis=0, ignore_index=True)

    labels_file = pd.read_csv(labels_csv_path)
    labels_file = labels_file[['ScanID', 'AgeYears']]
    labels_file.columns = ['scan_id', 'age']

    df = df.merge(labels_file, on = 'scan_id', suffixes=('',''))
    
    X = np.stack(df['data'])
    X = zscore(X, axis=2)
    labels = df[['scan_id', 'age']]
    
    print('X shape:', X.shape)
    print('y shape:', labels.shape)
    
    return X, labels
