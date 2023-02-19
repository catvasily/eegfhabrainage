# ---------------------------------------------------
# This is a copy of a script from Nikolai's README.md
# ---------------------------------------------------
from edf_preprocessing import PreProcessing

file_name = "81c0c60a-8fcc-4aae-beed-87931e582c45.edf"
#file_name = "81be60fc-ed17-4f91-a265-c8a9f1770517.edf"
#file_name = "fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2.edf"
#file_name = "ffff1021-f5ba-49a9-a588-1c4778fb38d3.edf"

path = "/data/eegfhabrainage/" + file_name
output_path = "/data/eegfhabrainage/results"

# Initiate the preprocessing object, filter the data between 0.5 Hz and 55 Hz and resample to 200 Hz.
p = PreProcessing(path, target_frequency=200, lfreq=0.5, hfreq=55)

# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
p.extract_good(target_length=60, target_segments=5)

# Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
p.save_edf(folder=output_path, filename=file_name)

# Extract and convert data to Numpy arrays
p.create_intervals_data()
df = p.intervals_df
