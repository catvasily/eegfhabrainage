import mne
import os
import os.path as op
import glob
from individual_func import save_notch_info, select_chans, set_channel_types, write_mne_edf,\
                            safe_crop
from mne.preprocessing import annotate_amplitude
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import zscore
import json
import re

JSON_CONFIG_FILE = "preproc_conf.json"
'''Default name (without a path) for the JSON file with preprocessing parameters. It is expected
to reside in the same folder as the *edf_preprocessing.py* source file.

'''

_JSON_CONFIG_PATHNAME = os.path.dirname(__file__) + "/" + JSON_CONFIG_FILE
'''Automatically generated fully qualified pathname to the default JSON config file

'''

def assign_known_channel_types(raw, *, conf_json = None,
                               conf_dict = None, ch_groups = None):
    '''Set channel types based on type information available in :data:`JSON_CONFIG_FILE`

    Args:
        raw (mne.Raw): the Raw object; channel data does not need to be preloaded
        conf_json (str): pathname of a json file with configuration parameters; in
           particular it must contain keys *"target_channels"* and *"exclude_channels"*.
           Default configuration file name is given by :data:`JSON_CONFIG_FILE` constant
        conf_dict (dict): a dictionary with configurartion parameters; in
           particular it must contain keys *"target_channels"* and *"exclude_channels"*.
           If both *conf_json* and *conf_dict* are given, the latter is used.
        ch_groups (dict): a dictionary with known channel types (see below). If None,
            this dictionary will be created

    Returns:
        ch_groups (dict): a dictionary with keys representing channel types and values as
            lists of corresponding channel names. If supplied as an argument,
            its keys (if present) will be updated.
    '''

    # Validate args
    if conf_json is None:
        conf_json = _JSON_CONFIG_PATHNAME

    if conf_dict is None:
        # Read configuraion from a json file
        with open(conf_json, "r") as fp:
            conf_dict = json.loads(fp.read())

    if ch_groups is None:
        ch_groups = dict()

    set_channel_types(raw, 'eeg', conf_dict["target_channels"], ch_groups = ch_groups)
    set_channel_types(raw, 'eog', conf_dict["eog_channels"], ch_groups = ch_groups)
    set_channel_types(raw, 'ecg', conf_dict["ecg_channels"], ch_groups = ch_groups)

    # Set all other channels to 'misc'
    # NOTE: one could set all channels to 'misc' from the beginning, then proceed
    # to setting known types. However MNE discards channel's units when setting 
    # the type to 'mics' - we do not want that for known channels
    known_lst = conf_dict["target_channels"].copy()
    known_lst.extend(conf_dict["eog_channels"])
    known_lst.extend(conf_dict["ecg_channels"])
    misc_lst = select_chans(raw.ch_names, known_lst, belong = False)[0]

    if misc_lst:
        set_channel_types(raw, 'misc', misc_lst, ch_groups = ch_groups)

    return ch_groups

def read_edf(filepath, *, conf_json = None, conf_dict = None, target_channels = None,
	exclude_channels = None, preload=True):
    '''
    Reads an EDF file with the MNE package, creates the Raw EDF object. 
    Excludes some channels to keep only target ones plus possibly some additional
    non-eeg channels. 

    All target channels are set to be 'eeg' type - that is, EEG sensor type
    Channels with names matching those in EOG, ECG channels lists are set to be 
    'eog', 'ecg' types respectively. All other included channels are set to be of 'misc' type.

    Raises an exception in case the input recording  doesn't have all 
    mandatory channels.
    
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
      preload (Bool): (default True) flag to read all the data into memory
    Returns:
      Raw (mne.Raw): raw EDF object

    Note:
        If both *conf_json* and *conf_dict* are omitted in the function call, the default settings will be read from 
       :data:`JSON_CONFIG_FILE` which should be present in the same folder as the *edf_preprocessing.py*.
       
    '''
    # Validate args
    if conf_json is None:
        conf_json = _JSON_CONFIG_PATHNAME

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
    # Read all channels except excluded ones without loading the data
    # Hopefully types of some channels will be inferred too
    raw = mne.io.read_raw_edf(filepath, exclude = exclude_channels,
                              infer_types=True,
                              verbose='warning',
                              preload = False)

    # Verify that all mandatory channels are present
    if select_chans(target_channels, raw.ch_names, belong = False)[0]:
        raise ValueError(filepath + ": File doesn't have all mandatory channels")

    # Now set types of known channels
    ch_groups = assign_known_channel_types(raw, conf_dict = conf_dict)

    if conf_dict["print_opt_channels"]:
        aux_list = select_chans(raw.ch_names, target_channels, belong = False)[0]

        if aux_list:
            aux_list.sort()
            print(filepath, " - additional channels included: ", aux_list)

    # Load channel's data into memory if requested
    if preload:
        raw.load_data()

    return raw

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
        notch_freq (float): notch frequency (typically - power line frequency)
        flat_parms (dict): parameters for flat intervals selection. Should contain the
            following keys:
            *'flat_max_ptp'* - the channel's amplitude max peak-to-peak value (in channel's
            units) for this channel to be marked as flat;
            *'bad_percent'* - min percentage of the time the channel's peak
            to peak is below the *'flat_max_ptp'* threshold to be considered flat;
            *'min_duration'* - minimum interval in seconds for all consecutive samples to
            be below the *'flat_max_ptp'* to indicate a flat interval.

    Note:
        * If both *conf_json* and *conf_dict*
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

    def __init__(self, filepath, *, conf_json = None, conf_dict = None,
                 target_channels = None, exclude_channels = None, target_frequency = None,
                 lfreq = None, hfreq = None, notch_freq = None, flat_parms = None):
        """Constructor args are explained in the class description above.
            
        """
        # Validate args
        if conf_json is None:
            conf_json = _JSON_CONFIG_PATHNAME

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

        if notch_freq is None:
            notch_freq = conf_dict["powerline_frq"]

        if flat_parms is None:
            flat_parms = conf_dict["flat_parms"]

        self.filename = filepath
        self.conf_dict = conf_dict
        self.target_channels = target_channels
        self.exclude_channels = exclude_channels
        self.target_frequency = target_frequency
        self.raw = read_edf(filepath, target_channels = target_channels, exclude_channels = exclude_channels,
                              preload = False)	# Do not read the data into memory just yet

        # Change non-standard channel names, if any
        # First find channels to rename
        to_rename = dict()
        for c in self.raw.ch_names:
            if c.upper() in conf_dict["rename_channels"]:
                to_rename[c] = conf_dict["rename_channels"][c.upper()]

        if to_rename:
            mne.rename_channels(self.raw.info, to_rename,
                                allow_duplicates=False, verbose=None)

        # Check if this EDF should not be processed for some reason
        self.skip_it = False	# Flag to advise to skip processing of this file
        duration = self.raw.times[-1] - self.raw.times[0]
        if duration > conf_dict['max_rec_length']:
            self.skip_it = True
            print('\n{} recording length is {} s which exceeds the max length threshold of {} s'.format(filepath,
                    duration, conf_dict['max_rec_length']))
        elif not conf_dict['allow_upsampling']:
            if self.raw.info['sfreq'] < target_frequency:
                self.skip_it = True
                print('\n{} sampling frequency {} Hz is smaller than the target frequency {} Hz'.format(filepath,
                    self.raw.info['sfreq'], target_frequency))

        if self.skip_it:
            print('The record was not loaded into memory, and no filtering or resampling was applied')
        else:
            self.raw.load_data()	# Read the full record now

            # Always notch powerline main frequency first (or whatever is notch setting)
            self.raw.notch_filter(freqs = notch_freq, picks = ['eeg', 'eog', 'ecg'], method='iir')

            # Save notch info with the raw object, as this is not done automatically
            save_notch_info(self.raw.info, notch_freq)

            self.raw.filter(l_freq=lfreq, h_freq=hfreq, picks = 'eeg', method = 'iir')
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
            intervals (list): [[start1, end1],...,[startN, endN]] or []: a list of start and end times
                of hyperventilation intervals.

        """
        hvticks = []	# Time stamps for annotations like "HV N min"
        # hvmins = []		# Minutes count for those time stamps
        ptrn = self.conf_dict["HV_regexp"]

        # Also find the end of HV marker, if any
        hv_end = None

        for ia, a in enumerate(self.raw.annotations.description):
            match = re.findall(ptrn, a)

            if len(match):
                hvticks.append(self.raw.annotations.onset[ia])
                # (dstart, dend) = re.search(r"\d+", a).span()
                # hvmins.append(int(a[dstart:dend]))

            if a.upper() == self.conf_dict["HV_end"]:
                hv_end = self.raw.annotations.onset[ia]

        if not hvticks:
            return []	# No HV intervals found

        hvticks = np.array(hvticks)

        if not (hv_end is None):
            hvticks = np.append(hvticks, hv_end)

        if len(hvticks) > 1:
            incs = hvticks[1:] - hvticks[:-1]
            gaps = incs > 90.	# If increment > 1.5 min - then this is a new HV interval 
        else:
            gaps = [False]

        pad_int = self.conf_dict["hv_pad_interval"]
        padd_it = lambda lst: [lst[0]-60.-pad_int, lst[-1]+60.+pad_int]

        if any(gaps):
            gaps = np.append(gaps, True)	# Make gaps same length as hvticks
            gaps = np.roll(gaps, 1)		# Now 0th element in gaps is always True

            lst_out = []
            l = [hvticks[0]]
            for i in range(len(gaps)):
                if i == 0:
                    continue
                if gaps[i]:
                    l.append(hvticks[i-1])	# Set the end of the previous interval
                    lst_out.append(l)		# Save the previous interval
                    l = [hvticks[i]]		# Start a new interval

            l.append(hvticks[-1])		# Set the end of the last interval
            lst_out.append(l)			# Save it

            # Padd all the intervals
            lst_final = []
            for l in lst_out:
                lst_final.append(padd_it(l))

            return lst_final		# Return multiple padded HV intervals
        # end of if any(gaps)

        return [padd_it(hvticks)]	# Return a single padded HV interval


    def photic_stimulation(self):
        """Identify beginning and end times of photic stimulation.
           
        Returns:
            intervals (list): [[start1, end1],...,[startN, endN]] or []: a list of start and end times
                of photic stimulation series.
             
        """
        starts = []	# Timestamps for annotationreferring to photic stim starts
        ends = []	# Timestamps for annotations referring to photic stim ends
        lst_starts = [s.upper() for s in self.conf_dict["photic_starts"]]
        lst_ends = [s.upper() for s in self.conf_dict["photic_ends"]]

        # loop over descriptions and identify those that contain frequencies
        for iannot, annot in enumerate(self.raw.annotations.description):
            a = annot.upper()

            # For stim start, the annotation must INCLUDE the keyword
            for kword in lst_starts:
                if kword in a:
                    # record the positions of stimulations
                    starts.append(self.raw.annotations.onset[iannot])
                    break

            # For stim end, the annotation must BE EQUAL TO the keyword
            # (because for example kword 'Off' may be part of other annotations - 
            # simply checking for its presence won't work)
            for kword in lst_ends:
                if kword == a:
                    # record the positions of stimulations
                    ends.append(self.raw.annotations.onset[iannot])
                    break

        # Avoid empty lists, to simplify processing.
        if not starts:
            # Insert one tick long interval at the start of the recording
            # (it will be discarded anyways)
            starts.append(self.raw.times[0])
            ends.insert(0, self.raw.times[1])

        # Sanity checks
        if ends[0] < starts[0]:
            starts.insert[0, 0.]	# Add start of stim at time = 0
            print('\nWARNING: Missing start of photic stimulation in file ', self.filename)
            print('Start of the recording is assumed\n')

        if ends[-1] < starts[-1]:
            ends.append(self.raw.times[-1])	# Add end of stim at time = end
            print('\nWARNING: Missing end of photic stimulation in file ', self.filename)
            print('End of the recording is assumed\n')

        # At this point always starts[0] < ends[-1]
        # Set up padding for any returned intervals
        pad_int = self.conf_dict["photic_pad_interval"]
        padd_it = lambda lst: [lst[0]-pad_int, lst[-1]+pad_int]	# Returns a single padded interval
                                                                # [tstart, tend]

        if len(starts) != len(ends):
            print('\nWARNING: Mismatch in numbers of start and end marks for photic stimulation in file ',
                    self.filename, '\n')
            return [padd_it([starts[0], ends[-1]])]

        # At this point len(starts) == len(ends). Convert starts and ends to arrays now
        starts = np.array(starts); ends = np.array(ends)

        if any(ends - starts < 0):
            print('\nWARNING: Some photic stim end marks precede start marks in file ',
                    self.filename, '\n')
            return [padd_it([starts[0], ends[-1]])]
            
        # Split photic stim into intervals, if there is more than one
        tmp = np.roll(starts, 1); tmp[0] =  starts[0]	# Shift starts to the right
        incs = starts - tmp				# Intervals between pulse onsets
        gaps = incs > self.conf_dict["max_isi"]

        if any(gaps):
            # Yes, there is more than one photic stim series
            gaps[0] = True			# We don't want to skip the 1st interval
            ser_starts = starts[gaps]		# Starting times of the series

            # One should use the ends which precede the new series starts
            # as the ends of previous series
            gaps1 = np.roll(gaps, -1); gaps1[-1] = True	# Shift gaps array one element left
            ser_ends = ends[gaps1]

            assert all(ser_ends - ser_starts > 0), "Starts and ends of photic stim intervals messed up"

            # Now numpy arrays ser_starts, ser_ends contain starts and ends of the series
            lst_out = []
            for i in range(len(ser_starts)):
                lst_out.append(padd_it([ser_starts[i], ser_ends[i]]))

            return lst_out
        else:
            # There is just one series
            return [padd_it([starts[0], ends[-1]])]

    def extract_good(self, target_length = None, target_segments = None):
        """This function calls the functions above to identify "bad" intervals and
        updates the attribute :data:`clean_intervals` with timesptamps to extract
        
        Args:
            target_length (float): length in seconds of the segments to be extracted
                from this EEG recording; the value from `self.conf_dict` will be used
                if not specified
            target_segments (int): a total number of the segments to extract 
                from this EEG recording; the value from `self.conf_dict` will be used
                if not specified

        Returns:
            None
             
        """
        if target_length is None:
            target_length = self.conf_dict["target_length"]

        if target_segments is None:
            target_segments = self.conf_dict["target_segments"]
        
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
            print('Found no clean intervals of the specified length in file ', self.filename)
            
    def save_edf(self, folder, filename):
        """ The function writes out new EDF file(s) based on :data:`clean_intervals` timestamps.
        It saves each segment into a separate EDF file, with suffixes "[scan_id]_1",
        "[scan_id]_2", etc. 
        
        Args:
            folder (str):    where to save the new EDF files
            filename (str) : main name for the output files (suffix will be added for more than 1 files)

        Returns:
            success (bool): `True` if clean interval(s) exist and are saved, `False` otherwise

        """
        
        # check if there are available clean segments
        if self.resolution:
            for i in range(len(self.clean_intervals)):
                interval_start = self.clean_intervals[i][0]
                interval_end = self.clean_intervals[i][1]
                
                tmp_raw_edf = self.clean_part.copy()
                safe_crop(tmp_raw_edf, interval_start, interval_end, include_tmax=False)
                
                if i > 0:
                    scan_id = filename.split('.')[0]
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+scan_id + '_' + str(i)+'.edf', overwrite=True)
                else:
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+filename, overwrite=True)

                return True
        else:
            print('Found no clean intervals of the specified length in file ', self.filename, '\n')
            return False
            
            
def slice_edfs(source_folder, target_folder, *, conf_json = None, conf_dict = None,
               source_scan_ids = None, target_frequency = None, lfreq = None, hfreq = None, 
               extract = 'good',
               target_length = None, target_segments = None, nfiles=None):
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
        source_scan_ids (list of str or None) : a list of short EDF file names without .edf
            extention to preprocess. If None, all .edf files in the source directory will
            be preprocessed, up to a limit set by the *nfiles* argument
        target_frequency (num or None): the final EEG frequency after resampling; if not specified
            the default for the :class: `PreProcessing` will be used 
        lfreq (float or None): the lower frequency boundary of the EEG signal in Hz; if not specified
            the default for the :class: `PreProcessing` will be used
        hfreq (float or None): the higher frequency boundary of the EEG signal in Hz; if not specified
            the default for the :class: `PreProcessing` will be used
        extract (str): one of 'good' (default) or 'HV'. In the first case good segments (no artifacts
            stimuli, etc. present). In the 2nd case hyperventilation intervals will be extracted.
        target_length (float): the length of each of the extracted segments in seconds; the value
            from `conf_json` or `conf_dict` will be used if not specified. Only applies when 
            `extract = 'good'`. In HV case the length will be equal to that of 1st HV interval.
        target_segments (int): the maximum number of segments to extract from each EDF file; the value
            from `conf_json` or `conf_dict` will be used if not specified. Only applies when 
            `extract = 'good'`. In HV case only the 1st HV interval will be extracted.
        nfiles (int or None): the max number of the source files to preprocess; (default = None = no limit)
          
    Returns:
        None

    """
    if conf_json is None:
        conf_json = _JSON_CONFIG_PATHNAME

    if conf_dict is None:
        # Read configuraion from a json file
        with open(conf_json, "r") as fp:
            conf_dict = json.loads(fp.read())

    if not (extract in ['good', 'HV']):
        raise ValueError("Unrecognized value for the 'extract' argument specified: extract = '{}'".format(extract))

    existing_edf_names = [op.basename(f) for f in glob.glob(source_folder + '/*.edf')]

    if source_scan_ids is None:
        scan_files = existing_edf_names
    else:
        scan_files = [scan_id if scan_id[-4:] == '.edf' else scan_id + '.edf' for scan_id in source_scan_ids]
        # scan_files = [scan_id + '.edf' for scan_id in source_scan_ids]

    cnt = 0
    print('\nProcessing files:')
    for f in scan_files:
        if f in existing_edf_names:
            path = source_folder + '/' + f
            print(f + '\t', end = '', flush = True)

            try:
                # Initiate the preprocessing object, resample and filter the data
                p = PreProcessing(path, conf_json = conf_json, conf_dict = conf_dict,
                                  target_frequency=target_frequency, lfreq=lfreq, hfreq=hfreq)

                if p.skip_it:
                    print("Record {} SKIPPED\n".format(f), flush = True)
                    continue

                #-------------------
                if extract == 'good':	# Extract good intervals
                #-------------------
                    # extract_good() calls member functions to detect 'bad intervals' and extract requested number of good
                    # segments of specified length
                    p.extract_good(target_length=target_length, target_segments=target_segments)
 
                    # Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
                    if p.save_edf(folder=target_folder, filename = f):
                        print('OK', flush = True)
                #-------------------
                elif extract == 'HV':
                #-------------------
                    lst_hv = p.hyperventilation()

                    if not lst_hv:
                        print('\nNo HV intervals found in the record\n')
                        continue

                    if len(lst_hv) > 1:
                        print('\nWarning: {} HV intervals found in the record.'.format(len(lst_hv)))
                        print('Only the first one is saved')

                    hv_int = lst_hv[0]

                    # Remove padding that was applied by hyperventilation() method
                    pad_int = conf_dict["hv_pad_interval"]
                    unpadd_it = lambda lst: [lst[0]+pad_int, lst[-1]-59.-pad_int]	# Extra second at the end
                    hv_int = unpadd_it(hv_int)

                    safe_crop(p.raw, hv_int[0], hv_int[1], include_tmax=False)	# !!! Raw object modified in place !!!

                    write_mne_edf(p.raw, fname=target_folder+'/'+f, overwrite=True)
                    print('OK', flush = True)
            
                cnt += 1
            except Exception as e:
                print('Record {} !!! FAILED !!!'.format(f))
                print(e, flush = True)
            
            if cnt % 100 == 0 and cnt != 0:
                print('\n{} EDFs processed\n'.format(cnt), flush = True)
                
            if cnt == nfiles:
                break
        else:	# File does not exist
            print('\n!!! File {} not found !!!\n'.format(f), flush = True)

    print('\nslice_edfs(): total of {} input EDF records processed.'.format(cnt), flush = True)
            
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
