from eeg_bpe import read_edf, preprocessing, define_bins, edf_to_str, edf2txt_pipeline, create_tokenizer, txt2df_pipeline, df2relative_pipeline
import os
import pandas as pd
from datetime import datetime

# Setting up pipeline parameters
EDF_NAMES = os.listdir("/home/mykolakl/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby/")[:10]

WS = 10 # window_size
NB = 20 # number of bins
MF = 100000 # tokenizer minimal frequncy
VS = 1500   # tokenizer vocabulary size

# all folders should already exist
TXT_FOLDER = './txt/txt_demo/'
EDF_FOLDER = '/home/mykolakl/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby/'
TARGET_FILE_NAME = 'demo'
TOKENIZER_PATH = './tokenizers/'
TOKENIZER_NAME = 'bpe_demo.json'

if not os.path.exists(TXT_FOLDER):
    os.makedirs(TXT_FOLDER)

# Convert EEG to symbolic series
edf2txt_pipeline(EDF_NAMES, EDF_FOLDER, TXT_FOLDER, WS=WS, NB=NB)
print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), "txt converted")

# Train tokenizer on symbolic series
tokenizer = create_tokenizer(TXT_FOLDER, VS, MF, 
                             TOKENIZER_PATH=TOKENIZER_PATH, 
                             TOKENIZER_NAME=TOKENIZER_NAME)
print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), "tokenizer trained")

# Encode symbolic series with traned tokenizer and count token frequencies
df = txt2df_pipeline(TXT_FOLDER, EDF_FOLDER, EDF_NAMES, TOKENIZER=tokenizer)
df.to_csv('dataframes/' + TARGET_FILE_NAME)

# Create dataframe of features based on 'relative' tokens
new_df = df2relative_pipeline(df, NB=NB) 
new_df.to_csv('dataframes/' + TARGET_FILE_NAME + ', relative')
print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), "dataframes created")
