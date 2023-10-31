import mne
import scipy
import scipy.stats
import numpy as np
import pywt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
from scipy.stats import kurtosis
from scipy.stats import skew
from sys import argv
import multiprocessing as mp


# Specify path of raw MEG data and output directory
PATH = "/home/hliaqat/scratch/MEG/"
OUT_PATH = "output/all_channels_exponweib_params/"
channels = ["MEG0113", "MEG0123", "MEG0133", "MEG0143", "MEG0213", "MEG0223", "MEG0233", "MEG0243", "MEG0313", "MEG0323", "MEG0333", "MEG0343", "MEG0413", "MEG0423", "MEG0433", "MEG0443", "MEG0513", "MEG0523", "MEG0533", "MEG0543", "MEG0613", "MEG0623", "MEG0633", "MEG0643", "MEG0713", "MEG0723", "MEG0733", "MEG0743", "MEG0813", "MEG0823", "MEG0913", "MEG0923", "MEG0933", "MEG0943", "MEG1013", "MEG1023", "MEG1033", "MEG1043", "MEG1113", "MEG1123", "MEG1133", "MEG1143", "MEG1213", "MEG1223", "MEG1233", "MEG1243", "MEG1313", "MEG1323", "MEG1333", "MEG1343", "MEG1413", "MEG1423", "MEG1433", "MEG1443", "MEG1513", "MEG1523", "MEG1533", "MEG1543", "MEG1613", "MEG1623", "MEG1633", "MEG1643", "MEG1713", "MEG1723", "MEG1733", "MEG1743", "MEG1813", "MEG1823", "MEG1833", "MEG1843", "MEG1913", "MEG1923", "MEG1933", "MEG1943", "MEG2013", "MEG2023", "MEG2033", "MEG2043", "MEG2113", "MEG2123", "MEG2133", "MEG2143", "MEG2213", "MEG2223", "MEG2233", "MEG2243", "MEG2313", "MEG2323", "MEG2333", "MEG2343", "MEG2413", "MEG2423", "MEG2433", "MEG2443", "MEG2513", "MEG2523", "MEG2533", "MEG2543", "MEG2613", "MEG2623", "MEG2633", "MEG2643"]
frequencies = [2, 6, 12, 24, 48]

# Input:
# subject: subject ID used for reading raw MEG file
# numSegments: Number of time segments to split the raw MEG file into
# segmentLength: How long in milliseconds each segment should be
# startTime/endTime: Used to specify start point and end point (in milliseconds) if not analyzing entire MEG
#                    If analyzing entire MEG, can remove from arguments and uncomment line 54, and comment line 53

# Output: Will save csv file for subject in directory specified in OUT_PATH
#         Returns string of subject upon completion
def create_rows(subject, numSegments, segmentLength, startTime, endTime):
    print("--------- Starting subject: " + str(subject) + "------------")

    global PATH
    global OUT_PATH
    global channels
    global frequencies
    
    # Read raw MEG file for subject
    raw_fname = PATH + "sub-" + subject + '/mf2pt2_sub-' + subject + '_ses-rest_task-rest_meg.fif'
    try:
        raw = mne.io.read_raw_fif(raw_fname)
    except:
        print("Couldn't read file for subject: " + subject)
        return

    # Create empty subject dataframe
    subjectdf = pd.DataFrame()

    # Iterate through each channel 
    for chan in channels:
        try:
            # Read time series from raw MEG
            Y = raw[chan][0][0][startTime:endTime]
            # y = raw[chan][0][0][:]
            z = scipy.stats.zscore(Y)
            
            for frequency in frequencies:
                print("calculating for channel: " + str(chan) + " and frequency: " + str(frequency))
                
                # Create sub-dataframe which will be appended to main subject dataframe
                df = pd.DataFrame()
                
                df['CCID'] = np.nan
                df.at[0, 'CCID'] = subject

                df['frequency'] = np.nan
                df.at[0, 'frequency'] = frequency

                df.at[0, 'channel'] = str(chan)
                
                # Continuous Wavelet Transform
                my_freq = np.arange(frequency, frequency+1)
                # Sampling frequency
                fs = 1000
                scale_range = fs*pywt.scale2frequency('cgau8', my_freq, precision=8)
                coef, freqs=pywt.cwt(z,scale_range,'cgau8',1/fs)
                
                # Calculate characetistics for each segment
                for segment in range(0, numSegments):
                    
                    # Create columns for recorded characteristics
                    df[str(segment + 1) + '_Mises_stat'] = np.nan
                    df[str(segment + 1) + '_skew'] = np.nan
                    df[str(segment + 1) + '_kurtosis'] = np.nan
                    df[str(segment + 1) + '_preidcted_param_1'] = np.nan
                    df[str(segment + 1) + '_preidcted_param_2'] = np.nan
                    df[str(segment + 1) + '_preidcted_param_3'] = np.nan
                    df[str(segment + 1) + '_preidcted_param_4'] = np.nan
                    
                    # y = np.absolute(coef[0][60000:300001])

                    # Extract specified time segment
                    startSeg = int(segment * segmentLength)
                    endSeg = int((segment+1) * segmentLength)
                    y = np.absolute(coef[0][startSeg:endSeg])
                    x = np.linspace(min(y), max(y), 200)

                    skew_ = skew(y, bias=False)
                    kurtosis_ = kurtosis(y, bias=False)
                    
                    # Fit distribution and test goodnest of fit using cramervonmises
                    distribution = "exponweib"
                    dist = getattr(scipy.stats, distribution)
                    param = dist.fit(y)
                    pdf_fitted = dist.pdf(x, *param)
                    frozen_dist = dist(*param)
                    res = scipy.stats.cramervonmises(y, frozen_dist.cdf)

                    # save characteristics
                    df.at[0, str(segment + 1) + '_Mises_stat'] = res.statistic
                    df.at[0, str(segment + 1) + '_skew'] = skew_
                    df.at[0, str(segment + 1) + '_kurtosis'] = kurtosis_

                    for i in range(0, len(param)):
                        df[str(segment + 1) + '_preidcted_param_' + str((i+1))] = np.nan
                        df.at[0, str(segment + 1) + '_preidcted_param_' + str((i+1))] = param[i]


                subjectdf = subjectdf.append(df)
        except Exception as e:
            print(e)
            print("Caught error for channel: " + str(chan) + " and frequency: " + str(frequency))
            continue
    print("Printing to csv: " + str(subject) + "_exponeweib_params.csv")
    subjectdf.to_csv(OUT_PATH + str(subject) + "_exponweib_params.csv", mode='a', header=False)
    return str(subject)


startTime = 0
#5min 30s
endTime = 330000
totalTime = endTime - startTime
numSegments = 11
segmentLength = totalTime/numSegments


subjects_list = []
# Read subject id's from a text file
with open('listfile.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        subjects_list.append(currentPlace)

# Multiprocessing with 48 cpus. Runs for 48 subjects at once, or by num_cpus specified in slurm file
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=48))
pool = mp.Pool(processes=ncpus)
results = [pool.apply_async(create_rows, args=(subject, numSegments, segmentLength, startTime, endTime,)) for subject in subjects_list]
subs = [p.get() for p in results]

print(subs)
print("----- done -----")
