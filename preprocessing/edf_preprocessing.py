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

def read_edf(filepath):
    '''
    Read an EDF file with MNE package, creates the Raw EDF object. 
    Excludes some channels to keep only 20 target ones.
    Prints warning in case the file doesn't have all 20 
    neeeded channels, doesn't return object in this case.
    
    Args:
      filepath: str with path to EDF file
    Returns:
      Raw EDF object
       
    '''
    # read EDF file while excluding some channels
    data = mne.io.read_raw_edf(filepath, exclude = ['A1', 'A2', 'AUX1', 
        'AUX2', 'AUX3', 'AUX4', 'AUX5', 'AUX6', 'AUX7', 'AUX8', 'Cz', 
        'DC1', 'DC2', 'DC3', 'DC4', 'DIF1', 'DIF2', 'DIF3', 'DIF4', 
        'ECG1', 'ECG2', 'EKG1', 'EKG2', 'EOG 1', 'EOG 2', 'EOG1', 'EOG2', 
        'Fp1', 'Fp2', 'Fpz', 'Fz', 'PG1', 'PG2', 'Patient Event', 'Photic', 
        'Pz', 'Trigger Event', 'X1', 'X2', 'aux1', 'phoic', 'photic'], verbose='warning', preload=True)
    
    # the list of target channels to keep
    target_channels = set(["FP1", "FPZ", "FP2", "F3", "F4", "F7", "F8", "FZ", "T3", "T4", "T5", "T6", "C3", "C4", "CZ", "P3", "P4", "PZ", "O1", "O2"])
    current_channels = set(data.ch_names)
    
    # checking whether we have all needed channels
    if target_channels == current_channels:
        return data
    else:
        print(filepath, "File doesn't have all needed channels")

class PreProcessing:
    """The class' aim is preprocessing clinical EEG recordings (in EDF format)
    and make them a suitable input for later analysis and ML applications.
     
    The class instantiates a preprocessing object which 
    carries a Raw EDF file through a sequence of operations: 
      
    #. resample each recording's signal to traget frequency
    #. keeps only signal with frequencies in selected range
    #. identifies timestamps of hyperventilation (HV), photic stimulation (PhS)
       and flat (zero) signal (together - "bad" intervals)
    #. extract EEG segment(s) of needed length from "good" intervals
         
    Then the object can save extracted segment(s) into new EDF files 
    OR return a Pandas DataFrame with data.
        
    Attributes:
        filename: string with EDF file path
        target_frequency: interger indicating the final EEG frequency after resampling
        raw: MNE Raw EDF object
        sfreq: initial sampling frequency of EEG
        bad_intervals: list of lists of the form [start, end] with timemstamps in seconds,
            indicating starts and ends of bad interval (HV, PhS, flat signal)
        clean_intervals: list of lists of the form [start, end] with timemstamps in seconds,
            indicating starts and ends of segments to be extracted
            
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

    def __init__(self, filepath, target_frequency, lfreq, hfreq):
        """

        Args:
            filepath: str with path to EDF file
            target_frequency: interger indicating the final EEG frequency after resampling
            lfreq: lower frequency boundary of the signal to keep
            hfreq: higher frequency boundary of the signal to keep
            
        """
        self.filename = filepath
        self.target_frequency = target_frequency
        self.raw = read_edf(filepath)
        self.raw.filter(l_freq=lfreq, h_freq=hfreq)
        self.sfreq = dict(self.raw.info)['sfreq']
        if(self.sfreq != self.target_frequency):
            self.raw.resample(self.target_frequency)
        
        self.clean_intervals = []
        self.intervals_df = pd.DataFrame()
        mne.set_log_level('warning')
        

    def flat_intervals(self):
        '''Identify beginning and end times of flat signal
        
        Returns:
            list of floats, contains start and end times
            
        '''
        annot_bad_seg, flat_chan = annotate_amplitude(self.raw, bad_percent=50.0, min_duration=10, flat=1e-06, picks=None, verbose=None)
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
            list of floats, contains start and end times
             
        """

        start = np.nan
        end = np.nan

        for position, item in enumerate(self.raw.annotations.description):
            if item in ["HV 1Min", "HV 1 Min"]:
                start = self.raw.annotations.onset[position] - 90
            if item in ["Post HV 30 Sec", "Post HV 60 Sec", "Post HV 90 Sec"]:
                end = self.raw.annotations.onset[position] + (90 - int(item.split(' ')[2]))

        if np.isnan(start):
            for position, item in enumerate(self.raw.annotations.description):
                if item in ["HV Begin", "Begin HV"]:
                    start = self.raw.annotations.onset[position] - 30

        if np.isnan(end):
            for position, item in enumerate(self.raw.annotations.description):
                if item in ["HV End", "End HV"]:
                    end = self.raw.annotations.onset[position] + 90

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
            # only ok when both starting and ending annotations are present. Is
            # it always the case? Added a check here:
            assert np.isnan(start) and np.isnan(end), \
                'Only one end of a hyperventilation interval is found'	# Assert when only one of them is NaN
            return []
        

    def photic_stimulation(self):
        """Identify beginning and end times of photic stimulation.
           
        Returns:
            list of floats, contains start and end times
             
        """
        
        # store times when stimulation occurs
        stimulation = []
        
        # loop over descriptions and identify those that contain frequencies
        for position, annot in enumerate(self.raw.annotations.description):
            if "Hz" in annot:
                # record the positions of stimulations
                stimulation.append(position)
        
        # provided stimulation has occured
        if len(stimulation)>1:
            
            # identify beginning and end
            start = self.raw.annotations.onset[stimulation[0]]
            end = self.raw.annotations.onset[stimulation[-1]] + self.raw.annotations.duration[stimulation[-1]]
            return [[start, end]]    
        else:
            return []
        
        # null value when no stimulation is present
        return None


    def extract_good(self, target_length, target_segments):
        """The function calls the functions above to identify "bad" intervals and
        updates the attribute clean_intervals with timesptamps to extract
        
        Args:
            target_length: length in seconds of the each 
                segments to extract from this EEG recording
            target_segments: number of segments to extract 
                from this EEG recording
             
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
        self.bad_intervals.insert(0,[0, 420]) # <--- TAKE FIRST SEVEN MINUTES AS BAD BY DEFAULT
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
        with the EEG data based on timestamps from clean_intervals.
        Print warning if no clean segments available.

        The DataFrame has the following columns:

        - scan_id - ID of the EEG recording.
        - interval_start - timestamp in datapoints of the segment start.
        - interval_length - length in datapoints of the segment.
        - data - numpy array of the EEG amplitude data, with shape
          (20, length in seconds X sampling frequency).
            
        Returns:
            self.interval_df - DataFrame with extracted segments
             
        NOTE: saving this dataframe into CSV file will truncate the content
        of 'data' column and convert it to string, so the data will be lost.
        Recommend to save data into .npy file separately and keep the .csv
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
            print('No clean intervals of needed length')
            
    def save_edf(self, folder, filename):
        """ The function write out new EDF file(s) based on clean_intervals timestamps.
        It save each segment into separate EDF file, with suffixes "[scan_id]_1",
        "[scan_id]_2", etc. 
        
        Args:
            folder - where to save new EDF files
            filename - main name for output files (suffix will be added for more > 1 files)

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
            print('No clean intervals of needed length')
            
            
def slice_edfs(source_scan_ids, source_folder, target_folder, target_frequency, 
               target_length, lfreq=0.5, hfreq=55, target_segments=1, nfiles=None):
    """ The function run a pipeline for preprocessing and extracting 
    clean segment(s) of needed length from multiple EDF files.
    It takes in list of EDF files names and prprocessing parameters, 
    look up for the files in source folder, and perform preprocessing 
    and extraction, if found.
    
    Args:
        source_scan_ids: list of EDF files to preprocess and extract segments from 
        source_folder: folder path with EDF files 
        target_folder: folder where to save extractd segments in EDF formats
        target_frequency: interger indicating the final EEG frequency after resampling
        target_length: length of each of the extracted segments (in seconds)
        lfreq: lower frequency boundary of the signal to keep in Hz (default=1)
        hfreq: higher frequency boundary of the signal to keep in Hz (default=55)
        target_segments: number of segments to extract from each EDF file;
            will extract less if less available (default=0.5)
        nfiles: limit number of files to preprocess and extract segments (default=None, no limit)
          
    """
   
    scan_files = [scan_id + '.edf' for scan_id in source_scan_ids]
    existing_edf_names = os.listdir(source_folder)

    i = 0
    
    for file in scan_files:

        if file in existing_edf_names:

            path = source_folder + '/' + file

            try:
                # Initiate the preprocessing object, resample and filter the data
                p = PreProcessing(path, target_frequency=target_frequency, lfreq=lfreq, hfreq=hfreq)

                # This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
                p.extract_good(target_length=target_length, target_segments=target_segments)

                # Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
                p.save_edf(folder=target_folder, filename=file)
            
                i += 1
                
            except:
                print('Extraction failed')
            
            if i % 100 == 0 and i != 0:
                print(i, 'EDF saved')
                
            if i == nfiles:
                break

            
def load_edf_data(source_folder, labels_csv_path):
    """The function loads multiple EDF files and returns data 
    with lables suitable for analysis and machine learning models.
    
    Args:
        source_folder: folder with EDF files
        labels_csv_path: CSV dile containing scan_ids and label (age)
        
    Returns:
        X (NumPy array): EEG amplitudes from EDF files, having
           shape of ([n_samples], 20, [length in seconds] x [sampling frequency])
        labels (NumPy array): age labels corresponding to each sample from X
        
    NOTE: this function fitted to a very specific CSV file 'age_ScanID.csv', 
    so it's looking for columns 'ScanID' for scan ids, 'AgeYears' for age labels. 
    So you might want to adjust column names in your file or change this function.
         
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
