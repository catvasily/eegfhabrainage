############################################################################################    
### Setup
from datetime import date
import sys
from string import ascii_lowercase
from scipy import signal
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal, stats
import pyedflib
from pyedflib import highlevel
from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta
import scipy
from scipy.stats import zscore
from collections import Counter
from pyts.approximation import PiecewiseAggregateApproximation
import pickle
from tokenizers import ByteLevelBPETokenizer
import dcor
import string

os.environ["TOKENIZERS_PARALLELISM"] = "true"

mne.set_log_level('warning')

import warnings
warnings.filterwarnings('ignore')


############################################################################################    
# Functions

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



def preprocessing(data, lowf, highf, window_size):
    """Filter frequencies betwee low and high values, 
    detrend signal, perform PAA aproximation on window size.
    
    Args:
        data : RawEDF instance
        low: low frequency limit
        high: high frequncy limit
        window_size: number of points to average signal on time axis
    
    Returns:
        Numpy array of filtered and aproximated signal with 20 channels
    """

    frequency = data.info['sfreq']
    if(frequency != 500):
        data.resample(500)
    
    data.filter(l_freq=lowf, h_freq=highf)
    
    np_data = data.get_data()
    np_data = zscore(np_data, axis=1)
    
    # Detrend signal data
    signal.detrend(np_data, axis=1, overwrite_data=True)
    
    # Set PAA, averaging series with WINDOW_SIZE
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    paa_data = paa.transform(np_data)
    
    return paa_data


def define_bins(low_fence, high_fence, n_bins):
    ''' This function devide the range of a value into N bins.
    
    Args:
      low_fence: lower outliers boundary
      high_fence: higher outliers boundary
      n_bins: number of bins to divide into
    Returns:
      List of bins margins    
    '''
    step = (high_fence - low_fence) / n_bins
    bins = []
    bins.append(low_fence)
    for b in range(n_bins):
        low_fence += step
        bins.append(low_fence)
    
    return bins



def edf_to_str(data, n_bins):
    """Perform several transformations: 
        1) define upper and lower limits for each array (1.5 * IQR)
        2) put values into N bins between these limits
        3) map bin values (1 .. N) to labels (a, b, ...)
    
    Args:
        data: nump array of eeg signal
        n_bins: number of bins to map values of signal between upper and lower 1.5 IQR 
    Returns:
        String of characters representing discretized EEG signal 
    """
    Q1,Q3 = np.percentile(data, [25,75])
    IQR = Q3 - Q1
    low_fence  = Q1 - (1.5 * IQR)
    high_fence = Q3 + (1.5 * IQR)
    
    bins = define_bins(low_fence, high_fence, n_bins)
    
    discrete_data = np.digitize(data, bins)
    
    if 'mapping' in locals():
        pass
    else:
        mapping = {}
        for key in range(n_bins + 2):
            mapping[key] = string.printable[key]
        
    str_data = ''.join([ mapping.get(item,item) for item in discrete_data ])
    
    return str_data


def save_metrics(y_test, y_predict):
    result = ''
    result += ('MAE:' + str(round(mae(y_test, y_predict), 3)) + '\n')
    result += ('Correlation:' + str(round(np.corrcoef(y_test, y_predict)[0,1], 3)) + '\n')
    result += ('Distance Correlation:' + str(round(dcor.distance_correlation(y_test, y_predict), 3)) + '\n')
    result += ('Explained variance:' + str(round(explained_variance_score(y_test, y_predict), 3)) + '\n')
    
    return result

def show_metrics(y_test, y_predict):
    ''' Plots a scatter plot of predicted age vs actual age with 
    fitted linear regression line to check model's preformance.
    An ideal model should return a plot, where all dots fit a
    45-degree line from the origin (bisector). Also displays following 
    metrics: mean absolute error, Pearson correlation, distnance 
    correlation and explained variance.
    
    Args:
        y_test: list of actual age
        y_predict: list of corresponding predicted age  
    '''
    print('MAE:', round(mae(y_test, y_predict), 3))
    print('Correlation:', round(np.corrcoef(y_test, y_predict)[0,1], 3))
    print('Distance Correlation:', round(dcor.distance_correlation(y_test, y_predict), 3))
    print('Explained variance:', round(explained_variance_score(y_test, y_predict), 3))
    
    plt.rcParams["figure.figsize"] = (10,10)
    plt.scatter(y_test, y_predict)
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    a, b = np.polyfit(y_test, y_predict, 1)
    plt.plot(y_test, a*y_test + b, 'r')

    plt.show()
    
############################################################################################        
## Multiple EDF's pipeline
#### Transforming EEG to TXT

############################################################################################    
##### EDFtoTXT - outliers defined per sample per channel

def edf2txt_pipeline(EDF_NAMES, EDF_FOLDER, TXT_FOLDER, WS, NB):
    ''' This function perform symbolization of EEG amplitude time series.
    The pipeline combines reading, cleaning and transorming EEG data
    (from EDF) files into TXT files (one per EDF file). It takes EDF files 
    from a source folder and save symbolyzed series as TXT files into target
    folder. Naming (ids) of files is preserved.
    
    Args:
        EDF_NAMES: list of filenames (ids) with in *.edf format
        EDF_FOLDER: path to a folder with corresponding EDF files
                    (should have / in the end)
        TXT_FOLDER: target folder where symbolic series will be saved as *.txt
        WS: number of points to average signal on time axis
        NB: number of bins to divide into
    '''

    for i in range(len(EDF_NAMES)):   

        try:
            file_name = EDF_NAMES[i]
            path = EDF_FOLDER + file_name

            edf_data = read_edf(path)

            paa_data = preprocessing(edf_data, lowf=0.5, highf=55, window_size=WS)

            str_data = []

            for c in range(paa_data.shape[0]):
                channel_data = paa_data[c, :]

                str_data.append(edf_to_str(channel_data, n_bins=NB))

            if len(str_data) < 20 or len(str_data) > 20:
                print(i, file_name, 'not all channels')
                pass

            with open(TXT_FOLDER + file_name[:-4] + '.txt', 'w') as f:
                f.write(','.join(str_data))

        except:
             print(i, file_name, 'failed')

            
#### Tokenization (Hugginface)

def create_tokenizer(TXT_FOLDER, VS, MF, TOKENIZER_PATH, TOKENIZER_NAME):
    ''' Train a Byte-Pair encoding tokenizer on a corpus of symbolic series. 
    
    Args:
        VS: vocabulary size, BPE tokenizer stops when reaching specified 
            tokens number in its vocabulary;
        MF: minimal frequency, tokenizer includes into vocabulary only tokens
            that appear specified number of times in the dataset, another
            stopping condition for training;
        TOKENIZER_PATH: folder where to save trained tokenizer
        TOKENIZER_NAME: name of tokenizer to save
    Returns:
        JSON file with vocabulary of trained tokenizer. The tokenizer can be
        restored (loaded) from this file.
    '''

    txts_path = glob.glob(TXT_FOLDER + '*.txt', recursive = True)

    #Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=txts_path, vocab_size=VS, min_frequency=MF, special_tokens=[","])

    tokenizer.save(TOKENIZER_PATH + TOKENIZER_NAME, pretty=True)
    
    return tokenizer


############################################################################################   
#### Encode multiple files with all channels, joined as columns:

def txt2df_pipeline(TXT_FOLDER, EDF_FOLDER, EDF_NAMES, TOKENIZER):
    ''' This function encodes symbolic series with pre-trained tokenizer and 
    count appearances of the token in each of the series. It returns a dataframe
    with EEG samples as rows and tokens as columns (different across channels) 
    
    Args:
        EDF_FOLDER: path to a folder with corresponding EDF files
                    (should have / in the end)
        TXT_FOLDER: target folder where symbolic series will be saved as *.txt
        TOKENIZER: tokenizer object
        
    Returns:
        df: pandas dataframe with scan_ids, token frequency features and "age" label
    '''

    file_names = os.listdir(TXT_FOLDER)
    
    # access a single EDF file just to get a list fo channels
    channel_names = read_edf(EDF_FOLDER + EDF_NAMES[0]).info['ch_names']

    df = pd.DataFrame()
    part_df = pd.DataFrame()
    i = 0

    for file_name in file_names:

        txt_path = TXT_FOLDER + file_name
        row_df = pd.DataFrame()

        with open(txt_path) as f:
            txt_file = f.readline()

        txt_channels = txt_file.split(',')

        for c in range(20):    

            tmp_output = TOKENIZER.encode(txt_channels[c])
            tmp_df = pd.DataFrame(Counter(tmp_output.tokens), index=[i]).divide(len(txt_channels[c])/100)
            tmp_df = tmp_df.add_prefix(channel_names[c].lower() + '_')

            row_df = pd.concat([row_df,tmp_df], axis=1, ignore_index=False)

        row_df['scan_id'] = file_name
        part_df = pd.concat([part_df,row_df], axis=0, ignore_index=True)

        #if i % 100 == 0:
            #print(i, 'folders packed')
            #print('tmp_df shape:', part_df.shape)

        if i % 500 == 0:
            df = pd.concat([df,part_df], axis=0, ignore_index=True)
            part_df = pd.DataFrame()
            #df.to_csv(TARGET_FILE_NAME)
            #print('df file saved', df.shape)

        i += 1

    df = pd.concat([df,part_df], axis=0, ignore_index=True)    
    df = df.fillna(0)

    # adjust frequency value to the token length
    for col in df.columns:
        if col == 'scan_id':
            df[col] = df[col].str.split('.', expand=True)[0]
        else:
            df[col] = df[col] * len(col.split('_')[1])
        
    # Get labels (age) and save df with absolute tokens
    labels = pd.read_csv('age_ScanID.csv')
    labels = labels[['ScanID', 'AgeYears']]
    labels.columns = ['scan_id', 'age']
    
    df = df.merge(labels, on = 'scan_id', suffixes=('',''))
    df = df.dropna()
    df['age'] = df['age'].astype('int32')
    
    return df


############################################################################################  
## Change tokens to relative differences

def df2relative_pipeline(df, NB): 
    ''' Change absolute tokens to realtive form. For example two tokens, 
    ‘bdc’ and ‘dfe’ have the same shape, but they are shifted across the
    amplitude axis. These two patterns of changes in EEG amplitude are 
    represented by different combinations of letters. We convert tokens 
    into their "relative" form by calculating the distance between the 
    adjacent symbols in terms of the number of bins between them. For
    example, in the token 'bdc', the distance between the letters 'b'
    and 'd' is two bins up, and that between 'd' and 'c' is one bin
    down. Thus, both tokens in had the same form [+2, -1]

    Args:
        df: pandas dataframe with symbolic form tokens, received from
            txt2df_pipeline
        NB: number of bins, that was used during symbolization of EEG
    
    '''
    n_bins = NB

    mapping = {}
    for key in range(n_bins + 2):
        mapping[string.printable[key]] = key

    tokens = list(df.drop(['scan_id', 'age'], axis=1).columns)

    rel_tokens = []

    for token in tokens:
        chan, shape = token.split('_') 
        rel_shape = []
        for i in range(1, len(shape)):
            current = mapping[shape[i]]
            prev = mapping[shape[i-1]]
            rel_shape.append(current-prev)

        rel_tokens.append(chan + '_' + str(rel_shape))

    uniq_rel_tokens = list(set(rel_tokens))

    token_map = pd.DataFrame({'token': tokens, 'rel_token': rel_tokens})

    rel_df = pd.DataFrame()
    rel_df['scan_id'] = df['scan_id']

    for rel_token in uniq_rel_tokens:
        if rel_token.split('_')[1] == '[]':
            pass
        else:
            corresponding_tokens = list(token_map[token_map['rel_token'] == rel_token]['token'])
            rel_df[rel_token] = 0
            for col in corresponding_tokens:        
                rel_df[rel_token] += df[col]

    # Get labels and save df with relative tokens
    labels = pd.read_csv('age_ScanID.csv')
    labels = labels[['ScanID', 'AgeYears']]
    labels.columns = ['scan_id', 'age']
    
    rel_df = rel_df.merge(labels, on = 'scan_id', suffixes=('',''))
    rel_df = rel_df.dropna()
    rel_df['age'] = rel_df['age'].astype('int32')

    return rel_df
