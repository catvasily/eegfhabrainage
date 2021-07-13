
import mne
import mxnet as mx
import os
import os.path as op
import glob
import numpy as np
import matplotlib.pyplot as plt 
from model import AGE_PREDICTION

def create_batch_input(path,batch_size=1,overlap=0,sampling_rate=512):
    '''
    batch_size : in seconds
    overlap : in seconds
    '''
    data = read_files(path)
    data = data.get_data()
    print(data.shape)
    batchs = []
    total_time = data.shape[1]/sampling_rate
    num_of_samples_per_batch = batch_size*sampling_rate
    num_of_batchs = int((total_time-overlap)/(batch_size-overlap))
    for i in range(num_of_batchs):
        if(overlap == 0):
            column_index = i * num_of_samples_per_batch
            batchs.append([data[:,column_index:column_index + num_of_samples_per_batch]])
        else:
            column_index = (i*(batch_size - overlap)) * sampling_rate
            batchs.append([data[:,column_index: column_index + num_of_samples_per_batch]])
    del data                    
    return np.array(batchs)


def read_files(filename):
    sample_data_raw_file = op.join(filename)
    raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
    raw = raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    raw = raw.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})
    raw = raw.set_channel_types({'PG1': 'eog', 'PG2': 'eog'})
    return raw

def main():
    # raw = read_files('../q1.EDF_folder/raw_good_afterPhotic.edf')
    # data = raw.get_data()

    model = AGE_PREDICTION()
    model.create_model()

    
    batch = create_batch_input('../q1.EDF_folder/raw_good_afterPhotic.edf')
    # batch = data[:,:512][:,:,np.newaxis]

    # batch = [data[:,:512]]
    print(batch.shape)
    label = []
    for i in range(batch.shape[0]): label.append(32)
    # a = []
    # a.append(batch)
    # a = mx.nd.array(a)
    a = mx.nd.array(batch)
    
    
    model.train(100,a,a,mx.nd.array(label))
    # print(a)

if __name__ == '__main__':
    main()