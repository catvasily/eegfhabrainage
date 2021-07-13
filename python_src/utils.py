
import mne
import os
import os.path as op
import glob
import numpy as np
import matplotlib.pyplot as plt 

def read_files(filename):
    sample_data_raw_file = op.join(filename)
    raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
    raw = raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    raw = raw.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})
    raw = raw.set_channel_types({'PG1': 'eog', 'PG2': 'eog'})
    return raw

def print_header(msg):
    print("--- {}".format(msg),"-"*80)

def plot_raw(raw):
    channels = raw.ch_names
    print_header("annotaiton")
    for i in raw.annotations.description:
        if('hz' in i): print(i)
    print_header("info")
    print(raw.info)
    print_header("channel names")
    print(channels)

    # print(mne.events_from_annotations(raw)[1])
    # raw.plot(start=0, duration=60, n_channels=3)
    # raw.plot_psd(average=True, dB=False)

    # print(dir(raw))
    # data = raw.get_data()
    # data = np.asarray(data)
    # print("data:",data.shape)
    # # # print(raw.pick('C3').get_data().shape)
    # plt.plot(data[2])
    # plt.show()
    

def main():
    raw = read_files('q1.EDF_folder/raw_good_afterPhotic.edf')
    plot_raw(raw)
    # raw = read_files('q1.EDF')
    # plot_raw(raw)

if __name__ == '__main__':
    main()