# import necessary modules
import glob
import os
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
import multiprocessing as mp
import os
import pyedflib
from pyedflib import highlevel
from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta


def read_files(file):
    """Read a EDF file and eliminate 14 out of 34 channels.

       Parameters
       ----------
       file : EDF file

       Output
       ----------
       data : truncated RawEDF with 20 channels

    """

    # read the file
    data = mne.io.read_raw_edf(file, exclude = ["Trigger Event", 
        "Patient Event", "ECG1", "ECG2", "AUX1", "AUX4", "AUX5", "AUX6", "AUX7",
        "AUX8", "AUX3", "PG1", "PG2", "A1", "A2", "EOG1", "EOG2", "EKG1", "EKG2", "AUX2", 
        "Photic", "phoic", "photic", "aux1"])

    return data


def frequency(data):
    """Check sampling frequency of a EDF file.

        Parameters
        ----------
        data : RawEDF instance

        Output
        ----------
        int, value of sampling frequency
    """
    # get information on file
    info = data.info

    # check sampling frequency
    return info['sfreq']


def photic_stimulation(data):
    """Identify beginning and end times of photic stimulation.

       Parameters
       ----------
       data : RawEDF instance

       Output
       ----------
       times : list of floats, contains start and end times
       """

    # store times when stimulation occurs
    stimulation = []

    # store beginning and ending times
    times = []

    # loop over descriptions and identify those that contain frequencies
    for position, annot in enumerate(data.annotations.description):
        if "Hz" in annot:
            # record the positions of stimulations
            stimulation.append(position)

    # provided stimulation has occured
    if len(stimulation)>1:

        # identify beginning and end
        start = data.annotations.onset[stimulation[0]]
        end = data.annotations.onset[stimulation[-1]] + data.annotations.duration[stimulation[-1]]
        times.extend([start, end])

    # list possibly empty if no photic stimulation occurs
    return times


def hyperventilation(data):
    """Identify beginning and end of hyperventilation from EEG data.

       Parameters
       ----------
       data : RawEDF instance

       Output
       ----------
       times : list of floats, contains start and end times
     """

    # labels to look for
    start_labels = ["HV Begin", "Hyperventilation begins", "Begin HV"]
    end_labels = ["HV End", "End HV", "end HV here as some fragments noted"]

    # parameters to check if hyperventilation is present
    # check for existence of start and end
    s = 0
    e = 0

    # store beginning and ending times
    times = []

    # identify start and end times of hyperventilation
    for position, item in enumerate(data.annotations.description):
        if item in start_labels:
            start = data.annotations.onset[position]
            s += 1
        if item in end_labels:
            end = data.annotations.onset[position] + data.annotations.duration[position]
            e += 1

    # when hyperventilation is present
    # eliminate the corresponding segment
    if s == 1 and e == 1:
        times.extend([start, end])

    if s ==2 or e ==2:
        return "Possibly bad file; manual check needed."

    # list possibly empty if no hyperventilation occurs
    return times


def cut_zeros(data):
    """Identify beginning and end of flat segments from EEG data.

       Parameters
       ----------
       data : RawEDF instance

       Output
       ----------
       intervals : list of lists of floats, each contains start and end times of a flat segment
       """

    # annotate segments that are flat
    # note : consider flat if duration is greater than 10 seconds
    annot_bad_seg, flat_chan = mne.preprocessing.annotate_flat(data, bad_percent=50.0, min_duration=10)

    # initialise lists for identification of flat segments
    starts = []
    durations = []
    bads = []
    intervals = []

    # loop over all new annotations
    for i in annot_bad_seg:

        # select beginning and duration of flat segment
        start = list(i.items())[0][1]
        duration = list(i.items())[1][1]

        # update the corresponding lists
        starts.append(start)
        durations.append(duration)
        bads.append('bads')
        end = start + duration
        intervals.append([start,end])

    # update file annotations
    data.set_annotations(data.annotations + mne.Annotations(starts, durations, bads, orig_time=data.info['meas_date']))

    # list possibily empty if there are no flat segments
    return intervals


def frequency_filter(x):
    """Filter frequencies betwenn 0.5 Hz and 100 Hz.

       Parameters
       ----------
       x : numpy ndarray
           The data structure to be filetred.

       Output
       ----------
       numpy array containing the filtered data
    """
    # store the filtered data
    filtered_list = []

    # import necessary modules
    from scipy import signal

    # create filter
    freq_filter = signal.firwin(500, [0.5, 100], pass_zero="bandpass", fs=512)

    # filter the data in all channels
    for i in range(len(x)):
        filtered_list.append(signal.convolve(x[i], freq_filter, mode='same'))

    return np.array(filtered_list)


def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def write_mne_edf(mne_raw, fname, picks=None, tmin=0, tmax=None,
                  overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')

    # static settings
    has_annotations = True if len(mne_raw.annotations)>0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607

    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])

    if tmin:
        date += timedelta(seconds=tmin)
    # no conversion necessary, as pyedflib can handle datetime.
    #date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None


    # convert data
    channels = mne_raw.get_data(picks,
                                start = first_sample,
                                stop  = last_sample)

    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []

        ch_idx = range(n_channels) if picks is None else picks
        keys = list(mne_raw._orig_units.keys())
        for i in ch_idx:
            try:
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': mne_raw._orig_units[keys[i]],
                           'sample_rate': mne_raw._raw_extras[0]['n_samps'][i],
                           'physical_min': mne_raw._raw_extras[0]['physical_min'][i],
                           'physical_max': mne_raw._raw_extras[0]['physical_max'][i],
                           'digital_min':  mne_raw._raw_extras[0]['digital_min'][i],
                           'digital_max':  mne_raw._raw_extras[0]['digital_max'][i],
                           'transducer': '',
                           'prefilter': ''}
            except:
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': mne_raw._orig_units[keys[i]],
                           'sample_rate': sfreq,
                           'physical_min': channels.min(),
                           'physical_max': channels.max(),
                           'digital_min':  dmin,
                           'digital_max':  dmax,
                           'transducer': '',
                           'prefilter': ''}

            channel_info.append(ch_dict)
        f.setPatientCode(mne_raw._raw_extras[0]['subject_info']['id'])
        #f.setPatientName(mne_raw._raw_extras[0]['subject_info']['name'])
        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)
        for annotation in mne_raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            description = annotation['description']
            f.writeAnnotation(onset, duration, description)

    except Exception as e:
        raise e
    finally:
        f.close()
    return True


# record if one segment can be selected from the file
good_list = []

def one_subject(item, scan_id):
    """Extract one minute from EEG data, with no hyperventilation, photic stimulation or flat segments."""

    # 0 -- good
    # 1 -- cannot read initial file
    # 2 -- cannot write the good minute
    # 3 -- the minute cannot be selected from the file


    # number of good segments identified
    n = 0

    try:
        # get the data in each file
        a = read_files(item)

    except:
        return 1

    # sampling frequency
    sfreq = int(frequency(a))

    # check for hyperventilation
    h = hyperventilation(a)
    if h:
        start_h, end_h = h

    # check for photic stimulation
    p = photic_stimulation(a)
    if p:
        start_p, end_p = p

    # check for flat segments
    f = cut_zeros(a)

    # combine the three lists
    f.append(h)
    f.append(p)

    # add beginning and end of file
    f.append([0, 0])
    f.append([a.times[-1], a.times[-1]])

    # sort sublists by first element
    f = sorted(f)

    # remove empty lists from f if exists
    f = list(filter(None, f))

    # loop over all segments
    for i in range(1, len(f)):

        # compute end of current segment
        # and start of the next one
        end_old = f[i-1][1]
        start_new = f[i][0]

        # when the difference between segments is greater than 5 min
        # we have found a good segment
        if start_new - end_old > 300:

            # mark good segment has been found
            n += 1

            # extract one minute after skipping 3 min
            b = a.copy()
            try:
                b.crop(tmin= end_old+180, tmax =end_old+241, include_tmax = False)
                write_mne_edf(b, '/project/6019337/cosmin14/1_minute/' + scan_id + '.edf', overwrite = True)
                return 0
            except:
                if os.path.exists('/project/6019337/cosmin14/1_minute/' + scan_id + '.edf'):
                    os.remove('/project/6019337/cosmin14/1_minute/' + scan_id + '.edf')
                return 2

    # no good segments exist
    if n == 0:
        return 3


def pre_processing(data):

    # extract the array
    x = data.get_data()

    # resample to 500 when necessary
    x = signal.resample(x, 500*60, axis = 1)

    # perform detrending
    signal.detrend(x, axis=1, overwrite_data=True)

    # normalize data
    x = stats.zscore(x, axis=1)

    # record the obtained data
    return x


# select filenames
df = pd.read_excel("/project/6019337/cosmin14/processed_metadata_2.xlsx")
filename = list(df['Filename'])
scan_id = list(df['ScanID'])
folder = glob.glob('/project/6019337/databases/eeg_fha/release_001/edf/Burnaby/*')


l = []
r = len(filename)
for i in range(6500, r):
    print(i+1)
    l.append(one_subject(filename[i], scan_id[i]))

df2 = pd.DataFrame({"Filename":filename[6500:], "Good/Bad": l})

df2.to_csv("/project/6019337/cosmin14/good.csv", mode = "a")
