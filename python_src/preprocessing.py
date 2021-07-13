"""
Created on Tue Oct 13 03:31:35 2020

@author: m_bon
"""

import mne
import os
import os.path as op
import glob
from individual_func import write_mne_edf
from mne.preprocessing import annotate_flat

# read file


def read_files(filename):
    sample_data_raw_file = op.join(filename)
    raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
    raw = raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    raw = raw.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})
    raw = raw.set_channel_types({'PG1': 'eog', 'PG2': 'eog'})
    return raw


# cut out 0s
def cut_zeros(raw):
    annot_bad_seg, flat_chan = annotate_flat(raw, bad_percent=50.0, min_duration=10,picks=None, verbose=None)
    raw_zeros = raw.copy()
    raw_zeros.crop(tmax=annot_bad_seg.onset[-1] + annot_bad_seg.duration[-1])
    write_mne_edf(raw_zeros,fname=os.path.join(os.getcwd(), filename + "_folder", 'raw_zeros.edf'),overwrite=True)
    return annot_bad_seg

# photic stimulation


def photic_stim(raw):
    matches = [s for s in raw.annotations.description if "Hz" in s]
    indexes_Hz = []

    for y in range(0, len(raw.annotations)-1):
        for x in matches:
            if x == raw.annotations.description[y]:
                indexes_Hz.append(y)

    print("\nphotic segment")
    raw_photic = raw.copy()
    raw_photic.crop(tmin=raw.annotations.onset[indexes_Hz[0]],tmax=raw.annotations.onset[indexes_Hz[-1]+1])
    write_mne_edf(raw_photic,fname=os.path.join(os.getcwd(), filename +"_folder", 'raw_photic.edf'),overwrite=True)
    print("done photic segment")

    return indexes_Hz

# remaining good segments


def remaining_good(raw, annot_bad_seg, indexes_Hz):

    print("\ngood segment before photic")
    raw_good = raw.copy()
    raw_good.crop(tmin=annot_bad_seg.onset[-1] + annot_bad_seg.duration[-1],
                  tmax=raw.annotations.onset[indexes_Hz[0]])
    write_mne_edf(raw_good,
                  fname=os.path.join(os.getcwd(), filename +
                                     "_folder", 'raw_good_beforePhotic.edf'),
                  overwrite=True)

    print("\ngood segment after photic")
    raw_good = raw.copy()
    raw_good.crop(tmin=raw.annotations.onset[indexes_Hz[-1]+1])
    write_mne_edf(raw_good,
                  fname=os.path.join(os.getcwd(), filename +
                                     "_folder", 'raw_good_afterPhotic.edf'),
                  overwrite=True)

    return

###


for filename in ['q1.EDF']:
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        newpath = os.path.join(os.getcwd(), filename + "_folder")
        if not os.path.exists(newpath):
            os.makedirs(filename + "_folder")
        raw = read_files(filename)
        annot_bad_seg = cut_zeros(raw)
        indexes_Hz = photic_stim(raw)
        remaining_good(raw, annot_bad_seg, indexes_Hz)

# def main():


# if __name__ == '__main__':
