import mne
import os
import os.path as op
import glob
from individual_func import write_mne_edf
from mne.preprocessing import annotate_flat
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import zscore

def read_edf(filepath):
    data = mne.io.read_raw_edf(filepath, exclude = ["Trigger Event", 
    "Patient Event", "ECG1", "ECG2", "AUX1", "AUX4", "AUX5", "AUX6", "AUX7",
    "AUX8", "AUX3", "PG1", "PG2", "A1", "A2", "EOG1", "EOG2", "EKG1", "EKG2", "AUX2", 
    "Photic", "phoic", "photic", "aux1"], verbose='warning', preload=True)

    return data


class PreProcessing:

    def __init__(self, filepath, target_frequency):
        self.filename = filepath
        self.target_frequency = target_frequency
        self.raw = read_edf(filepath)
        self.raw.filter(l_freq=0.5, h_freq=55)
        self.sfreq = dict(self.raw.info)['sfreq']
        if(self.sfreq != self.target_frequency):
            self.raw.resample(self.target_frequency)
        
        self.clean_intervals = []
        self.intervals_df = pd.DataFrame()
        mne.set_log_level('warning')
        

    def flat_intervals(self):
        '''Identify beginning and end times of flat signal
        Output
        ----------
        list of floats, contains start and end times
        '''
        annot_bad_seg, flat_chan = annotate_flat(self.raw, bad_percent=50.0, min_duration=10,picks=None, verbose=None)
        intervals = []
        for i in annot_bad_seg:
            start = list(i.items())[0][1]
            duration = list(i.items())[1][1]
            end = start+duration
            intervals.append([start,end])
        return intervals


    def hyperventilation(self):
        """Identify beginning and end of hyperventilation from EEG data"""
        
        # labels to look for
        start_labels = ["HV Begin", "Hyperventilation begins", "Begin HV"]
        end_labels = ["HV End", "End HV", "end HV here as some fragments noted"]
        
        # parameters to check if hyperventilation is present
        # check for existence of start and end
        s = 0
        e = 0
        
        # identify start and end times of hyperventilation
        for position, item in enumerate(self.raw.annotations.description):
            if item in start_labels:
                start = self.raw.annotations.onset[position]
                s += 1
            if item in end_labels:
                end = self.raw.annotations.onset[position] + self.raw.annotations.duration[position]
                e += 1
        
        # when hyperventilation is present
        # eliminate the corresponding segment
        if s == 1 and e == 1:
            return [[start, end]]
        else:
            return []
        
        if s ==2 or e ==2:
            return "Possibly bad file; manual check needed."
        
        # null value when no hyperventilation is present
        return None

    def photic_stimulation(self):
        """Identify beginning and end times of photic stimulation.
            
        Parameters
        ----------
        self.raw : RawEDF instance
        
        Output
        ----------
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


    def extract_good(self, target_length, target_slices):
        self.bad_intervals = []
        self.bad_intervals.extend(self.flat_intervals())
        self.bad_intervals.extend(self.hyperventilation())
        self.bad_intervals.extend(self.photic_stimulation())
        self.bad_intervals.sort()
        #print(bad_intervals)
        
        self.clean_part = self.raw.copy()
        tmax = len(self.raw)/self.target_frequency
                
        # Add 'empty' bad intervals in the begging and in the end for furhter consistency
        self.bad_intervals.insert(0,[0, 60]) # <--- TAKE FIRST MINUTE AS BAD BY DEFAULT
        self.bad_intervals.append([tmax, tmax])
        # Construct temporary dataframe to find clean interval in EDF
        tmp_df = pd.DataFrame(self.bad_intervals, columns=['start', 'end'])
        
        # Define end of the clean interval as a start of next bad interval
        tmp_df['next_start'] = tmp_df['start'].shift(periods=-1)
        tmp_df.iloc[-1,-1] = tmax # <= Assign end of edf file as the end of last clean interval
        
        # Handle cases when earlier bad interval overlap next intervals
        prev_value = 0
        new_ends = []
        for value in tmp_df['end']:
            if prev_value > value :
                new_ends.append(prev_value)
            else:
                new_ends.append(value)
                prev_value = value
        tmp_df['cumulative_end'] = new_ends
        
        # Calculate lengths of clean intervals
        tmp_df['clean_periods'] = tmp_df['next_start'] - tmp_df['cumulative_end']
        
        # Check whether there is at least 1 clean interval with needed target length
        if tmp_df[tmp_df['clean_periods'] >= target_length].shape[0] == 0:
            self.resolution = False
            pass
        else:    
            self.resolution = True
    
            total_available_slices = (tmp_df[tmp_df['clean_periods'] > 0]['clean_periods'] // target_length).sum()
        
            if target_slices < total_available_slices:
                n_samples = target_slices
            else:
                n_samples = total_available_slices
                
            starts = list(tmp_df[tmp_df['clean_periods'] > 0]['cumulative_end'])
            n_available_slices = list(tmp_df[tmp_df['clean_periods'] > 0]['clean_periods'] // target_length)
            #print('n_available_slices', n_available_slices)
                
            for i in range(len(n_available_slices)):
                current_start = int(starts[i])
                #print(i, current_start)
                for s in range(int(n_available_slices[i])):
                    #print(s, current_start)
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
        if self.resolution:

            ids = np.repeat(self.filename.split('/')[-1].split('.')[0], len(self.clean_intervals))
            intervals_data = []
            interval_starts = []
            interval_lengths = []
            channels = []
            
            for i in range(len(self.clean_intervals)):
                   
                interval_start = self.clean_intervals[i][0] * self.target_frequency
                interval_end = self.clean_intervals[i][1] * self.target_frequency
                
                interval_data = self.clean_part.get_data(start=interval_start, stop=interval_end)
                interval_data = zscore(interval_data, axis=1)
                
                intervals_data.append(interval_data)
                interval_starts.append(interval_start)
                interval_lengths.append(interval_end - interval_start)
                
            self.intervals_df['scan_id'] = ids
            self.intervals_df['interval_start'] = interval_starts
            self.intervals_df['interval_length'] = interval_lengths
            self.intervals_df['data'] = intervals_data
        else:
            print('No clean intervals of needed length')
            
    def save_clean_part(self, folder, filename):
        if self.resolution:
            for n in range(len(self.clean_intervals)):
                interval_start = self.clean_intervals[n][0]
                interval_end = self.clean_intervals[n][1]
                
                tmp_raw_edf = self.clean_part.copy()
                
                tmp_raw_edf.crop(interval_start, interval_end, include_tmax=True)
                
                if n > 0:
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+str(n)+'_'+filename, overwrite=True)
                else:
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+filename, overwrite=True)
        else:
            print('No clean intervals of needed length')
