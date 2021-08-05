import mne
import os
import os.path as op
import glob
from individual_func import write_mne_edf
from mne.preprocessing import annotate_flat
from utils import plot_raw, plot_data  
import matplotlib.pyplot as plt 
import numpy as np

from DataProvider import DataProvider

class PreProcessing:

    def __init__(self,filename):
        self.filename = filename
        self.raw = DataProvider.read_files(filename)
        self.sfreq = dict(self.raw.info)['sfreq']
        if(self.sfreq != 500):
            self.resample()

    def cut_zeros(self):
        '''Identify beginning and end times of flat signal

        Output
        ----------
        list of floats, contains start and end times

        '''

        print('-'*40,"Extracting Flat",'-'*40)
        annot_bad_seg, flat_chan = annotate_flat(self.raw, bad_percent=50.0, min_duration=10,picks=None, verbose=None)
        intervals = []
        for i in annot_bad_seg:
            start = list(i.items())[0][1]
            duration = list(i.items())[1][1]
            end = start+duration
            intervals.append([start,end])
        return intervals


    def hyperventilation(self):
        print('-'*40,"Extracting Hyperventilation",'-'*40)
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
        print('-'*40,"Extracting Photic Stimulation",'-'*40)
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


    def extract_good(self):
        bad_intervals = []
        bad_intervals.extend(self.cut_zeros())
        bad_intervals.extend(self.hyperventilation())
        bad_intervals.extend(self.photic_stimulation())
        print('-'*40,"Extracting One Minute",'-'*40)
        # bad_intervals.append([13,1600])
        bad_intervals.sort()
        print(bad_intervals)
        disjoint_bad_intervals = []
        i = 0
        while(i<len(bad_intervals)):
            s = bad_intervals[i][0]
            e = bad_intervals[i][1]
            while(i<len(bad_intervals) and e>=bad_intervals[i][0]):
                e = max(e,bad_intervals[i][1])
                i+=1

            disjoint_bad_intervals.append([s,e])

        self.one_min = self.raw.copy()
        tmax = len(self.raw)/self.sfreq
        for i, val in enumerate(disjoint_bad_intervals):
            if(i==0):
                if(val[0]>60):
                    s = (val[0]-60)/2
                    e = s+60
                    self.one_min.crop(s,e, include_tmax=True)
                    break

                nxt_start = tmax
                if(i+1<len(disjoint_bad_intervals)): 
                    nxt_start = disjoint_bad_intervals[i+1][0]
                if(nxt_start-val[1] > 60):
                    s = val[1] + (nxt_start - val[1] - 60)/2
                    e = s + 60
                    self.one_min.crop(s,e, include_tmax=True)
                    break
            elif(i==len(disjoint_bad_intervals)-1):
                if(tmax - val[1] > 60):
                    s = val[1] + (tmax - val[1] - 60)/2
                    e = s+60
                    self.one_min.crop(s,e, include_tmax=True)
                    break
        
            else:
                if(disjoint_bad_intervals[i+1][0]-val[1] > 60):
                    s = val[1] + (disjoint_bad_intervals[i+1][0] - val[1] - 60)/2
                    e = s + 60
                    self.one_min.crop(s,e, include_tmax=True)
                    break
    
        print("One minute segment (start,end) == ","({},{})".format(s,e))          
        print(self.one_min.get_data().shape)
        print(disjoint_bad_intervals)

    def save_one_min(self,path):
        write_mne_edf(self.one_min,fname=path,overwrite=True)

    def resample(self):
        self.raw.filter(l_freq=0.5, h_freq=100)
        self.raw.resample(500)



if __name__ == '__main__':
    filename = 'b1.edf'
    p = PreProcessing(filename)
    p.extract_good()
    p.save_one_min('onemin.edf')
    one_min = DataProvider.read_files('onemin.edf')
    print(one_min.get_data().shape)