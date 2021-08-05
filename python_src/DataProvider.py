import mne
import os
import os.path as op
import glob
from individual_func import write_mne_edf
from mne.preprocessing import annotate_flat
from utils import plot_raw, plot_data  
import matplotlib.pyplot as plt 
import numpy as np

class DataProvider:

    def __init__(self):
        pass

    @staticmethod
    def read_files(filename):
        sample_data_raw_file = op.join(filename)
        raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
        raw = raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        # raw = raw.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})
        raw = raw.set_channel_types({'PG1': 'eog', 'PG2': 'eog'})
        return raw


if __name__ == '__main__':
    a = DataProvider.read_files('onemin.edf')
    print(a.get_data().shape)