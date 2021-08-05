import mne
import os
import os.path as op
import glob
from individual_func import write_mne_edf
from mne.preprocessing import annotate_flat
from utils import plot_raw, plot_data  
import matplotlib.pyplot as plt 
import numpy as np
# from utils import create_batch_input

def mark_bad_channels(raw):
    raw.info['bads'] = ['A1','A2','AUX2','AUX5','AUX6','AUX7','AUX8','PG1','PG2']


def read_files(filename):
    sample_data_raw_file = op.join(filename)
    raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
    raw = raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    # raw = raw.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})
    raw = raw.set_channel_types({'PG1': 'eog', 'PG2': 'eog'})
    return raw

def cut_zeros(raw):
    mark_bad_channels(raw)
    annot_bad_seg, flat_chan = annotate_flat(raw, bad_percent=50.0, min_duration=10,picks=None, verbose=None)
    starts = []
    durations = []
    bads = []
    intervals = []
    for i in annot_bad_seg:
        start = list(i.items())[0][1]
        duration = list(i.items())[1][1]
        starts.append(start)
        durations.append(duration)
        bads.append('bads')
        intervals.append([start,end])
        print(i)
    print(" --- Annotaions before ----:", raw.annotations)
    raw.set_annotations(raw.annotations + mne.Annotations(starts, durations, bads, orig_time=raw.info['meas_date']))
        # flats.append(start)
    
    for ch in flat_chan:
        raw.info['bads'].append(ch)
    #     end = start+duration
    # data = raw.get_data()
    # for ch in data:
    #     del ch[int(start*500): int(end*500)]
    # return data
    print(" --- Annotaions after ----:",raw.annotations)
    return intervals

def photic_stim(raw):
    matches = [s for s in raw.annotations.description if "Hz" in s]
    indexes_Hz = []

    for y in range(0, len(raw.annotations)-1):
        for x in matches:
            if x == raw.annotations.description[y]:
                indexes_Hz.append(y)
    return indexes_Hz

def clean_data(raw):
    annot_bad_seg = cut_zeros(raw)
    indexes_Hz = photic_stim(raw)
    try:
        raw.crop(tmin=annot_bad_seg.onset[-1] + annot_bad_seg.duration[-1],
                    tmax=raw.annotations.onset[indexes_Hz[0]])
        raw.crop(tmin=raw.annotations.onset[indexes_Hz[-1]+1])
    except:
        pass

def extract_one_minute(raw):
    data = raw.get_data()
    start = int((data.shape[1]- (60*500))/2)
    end = start + (60*500)
    one_min = []
    for ch in data:
        one_min.append(ch[start:end])
    return np.array(one_min)

def resample(raw):
    raw.filter(h_freq=100)
    raw.resample(500)

if __name__ == '__main__':
    raw = read_files('b1.edf')
    # print(dict(raw.info)['secs'])
    print(len(raw))
    print(raw.get_data().shape)
    # print(raw.annotations)
    # # plot_raw(raw)
    # # clean_data(raw)
    # plot_raw(raw)
    # # plt.show()    
    # # data = raw.load_data()
    # print(raw.ch_names)
    # flats = cut_zeros(raw)
    # a = extract_one_minute(raw)
    # print(raw.info)
    # print(a.shape)
    # # print(raw.info)
    # # raw.plot()
    # # print(raw.info)
    # # print(data.shape)
    # # plot_raw(raw)
    # plt.show()