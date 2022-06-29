# EEG preprocessing pipeline for FHA EDF dataset

## Summary

The code is aimed to preprocess clinical EEG recordings and make them a suitable input for later analysis and ML applications. 
- **Input**: EDF file
- **Output**: 
  - (1) Pandas DataFrame with columns: file name, interval start, interval end, and data - Numpy array of shape (20, 500 * target length) 
  - (2) new EDF file(s) containing only clean EEG interval of target length

**Parameters**:
- number of clean EEG slices to extract from each EDF file
- target length of each slice, in seconds

## Performed operations
1) resample each recording's signal to 500 Hz frequency, since some recordings might have different sampling frequencies. 
2) applied frequency filtering that keeps only signal with frequencies between 0.5 and 55 Hz, to exclude unwanted noise such as electricity grid frequency (60 Hz in Canada) or sudden patients` moves (<1 Hz)
3) identify and remove intervals of special procedures performed on patients during recordings, such as hyperventilation (deep breathing) and photic stimulation (flashing light). Physicians apply these tests to patients in order to detect brain abnormal activity for epilepsy diagnosis. Since these procedures burst abnormal activity, and weren't performed for all subjects, we exclude them from the analysis. Also the recordings contain intervals with no signal. It is the results of turned off equipment or disconnected electrode. So we also have to avoid these flat intervals with zero signal. Thus traget slices acquired only from clean intervals from each EEG, without flat intervals, hyperventilation and photic stimulation. Slices taken from the beginning, first minute taken as "bad" by default. The algoritm also handles cases when "bad intervals" overlap
4) **In case of extracting to Numpy arrays** signal values are also ZScore noramilized. Doesn't apply in case of saving output to EDF file(s).


## Usage
You need both modules edf_preprocessing.py and individual_func.py. The later contains python routine for saving output in EDF format again.

```python

from edf_preprocessing import PreProcessing

file_name = "81c0c60a-8fcc-4aae-beed-87931e582c45.edf"
path = "/home/mykolakl/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby/" + file_name
output_path = "your_folder"

# Initiate the preprocessing object, resample and filter the data
p = PreProcessing(path, target_frequency=200)

# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
p.extract_good(target_length=60, target_slices=5)

# Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
p.save_clean_part(folder=output_path, filename=file_name)

# Extract and convert data to Numpy arrays
p.create_intervals_data()
df = p.intervals_df

```

**NOTE**
When extracting to Numpy, saving this dataframe to a csv file will damage your data. 
It converts Numpy arrays to string and truncate it, leaving few values in the beginning and the end.
    
To save your data in numpy arrays format for later use, you can use this code:
```python
numpy_data = np.stack(df['data'])

np.save(output_path, numpy_data)
```

As the result, you will get numpy array of shape (number_of_slices, 20, lengths_in_seconds * 500). Following the above example it is (5, 20, 30000).
You should also save inital df to csv to keep track of scan IDs for later match with target label, like age.


## Multiple files pipeline

Here is jsut an example of loop that you can use to slice multiple files. The output of these routine will be multiple NumPy array and csv files: 

```python
import os
import pandas as pd
import numpy as np
from edf_preprocessing import PreProcessing

# list of EDF files names, here you need to specify your folder
edf_names = os.listdir('/home/[your_folder]/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby')

# number of EDF files which slices will saved in a single .npy file
chunk_size = 500

df = pd.DataFrame()

# iterating through files
for i in range(len(edf_names)):   
    
    file_name = edf_names[i]
    path = f"/home/[your_folder]/projects/rpp-doesburg/databases/eeg_fha/release_001/edf/Burnaby/" + file_name

    p = PreProcessing(path, target_frequency=200)
    try:
        p.extract_good(target_length=60, target_slices=5)
        p.create_intervals_data()
        df = pd.concat([df, p.intervals_df], axis=0, ignore_index=True)
    except:
        print('Cleaning failed')
               
    if i % 100 == 0:
        print(i, 'EDF extracted')
        
    if i % chunk_size == 0 and i != 0:
        edf_npdata = np.stack(df['data'])
        np.save('eeg_slices_1min/np_data_' + str(int(i/chunk_size)), edf_npdata)
        
        df.to_csv('eeg_slices_1min/scan_ids_' + str(int(i/chunk_size)) + '.csv')
        
        df = pd.DataFrame()

edf_npdata = np.stack(df['data'])

#saving files with suffixes like 'np_data_1', 'np_data_2', etc
np.save('eeg_slices_1min/np_data_' + str(int(i/chunk_size) + 1), edf_npdata)
df.to_csv('eeg_slices_1min/scan_ids_' + str(int(i/chunk_size) + 1) + '.csv')
```
