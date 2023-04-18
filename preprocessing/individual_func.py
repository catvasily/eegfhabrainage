# -*- coding: utf-8 -*-
"""
saving the file as .edf

Created on Wed Dec  5 12:56:31 2018
@author: skjerns

Gist to save a mne.io.Raw object to an EDF file using pyEDFlib
(https://github.com/holgern/pyedflib)

Disclaimer:

- Saving your data this way will result in slight 
  loss of precision (magnitude +-1e-09).
- It is assumed that the data is presented in Volt (V), 
  it will be internally converted to microvolt
- BDF or EDF+ is selected based on the filename extension
- Annotations are lost in the process.

"""

"""
Let me know if you need them, should be easy to add.

"""

import pyedflib # pip install pyedflib
from pyedflib import highlevel # new high-level interface
from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta
import mne
import os
import numpy as np
import re

import warnings

def select_chans(ch_list, target_list, belong = True):
    """ From the input channel list `ch_list` select channels that belong to the target list.
    The string comparison is performed case insensitive, but original case is
    preserved in the returned list.

    Args:
        ch_list (list of str): input channel list
        target_list (list of str): a target channel list
        belong (bool): True (default) if request is to find channels that belong to the target list,
            False if one wants channels that do NOT belong to the target list

    Returns:
        selected_channels, flags
        selected_channels (list of str): a list of channels from the input list
            that belong to the `target list`
        flags (list of bool): list of flags indicating if elements of `ch_list`
            belong / not belong to the `target_list`; `len(flags)` equals to `len(ch_list)`
    """

    target_upper  = [s.upper() for s in target_list]
    
    selected = []
    flags = []
    yes = lambda x: belong if x.upper() in target_upper else (not belong)

    for item in ch_list:
        if yes(item):
            selected.append(item)
            flags.append(True)
        else:
            flags.append(False)

    return selected, flags

def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def set_channel_types(raw, type_name, type_list, ch_groups = None):     
    '''Set data channel types based on their names.

    If a channel name belongs to the `type_list` (case-insensitive),
    then corresponding channel's type will be set to the `type_name`.
    The raw object will also be updated. As a result, 
    `ch_groups[type_name]` will contain all channels belonging to
    the type including those that were previously marked in self.raw.
    

    Args:
        raw (mne.Raw): the Raw object; channel data does not need to be preloaded
        type_name (str): the channel type, such as 'eog', 'ecg', 'eeg', etc.
        type_list(list of str): list of known channel names that belong to this type.
        ch_groups (dict): a dictionary with known channel types (see below). If None,
            this dictionary will be created
    Returns:
        ch_groups (dict): a dictionary with keys corresponding to channel types and
            values being lists of corresponding channel names. If supplied as an argument,
            its key equal to `type_name` will be updated.
    '''
    # Get channels that already have the target type
    raw1 = raw.copy()
    kwargs = {type_name: True}

    if ch_groups is None:
        ch_groups = dict()

    try:
        raw1.pick_types(**kwargs)
        ch_groups[type_name] = raw1.ch_names
    except ValueError:
        ch_groups[type_name] = []

    del raw1

    tlst, _ = select_chans(raw.ch_names, type_list)

    if tlst:
        mapping = dict.fromkeys(tlst, type_name)

        # Ignore warning when changing units to NA for misc channels
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message = r"The unit for channel.+\shas changed from V to NA")
            raw.set_channel_types(mapping)

        l = ch_groups[type_name]		# Current list
        l.extend(tlst)				# Append found channels
        ch_groups[type_name] = list(set(l))	# Remove dupes, if any

    return ch_groups

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
    
    #print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']

    # Prepare 'prefilter' string for the channel_info
    prefilter = ''
    if not np.isclose(mne_raw.info['highpass'], 0.):
        prefilter = 'HP:{}Hz'.format(mne_raw.info['highpass'])

    if not np.isclose(mne_raw.info['lowpass'], sfreq/2):
        if len(prefilter) > 0:
            prefilter = prefilter + ' '

        prefilter = prefilter + 'LP:{}Hz'.format(mne_raw.info['lowpass'])

    # Add notch info, if any
    if not (mne_raw.info['description'] is None):
        tmp = re.search(r"Notch filter: \d+\.*\d*\s*Hz",
                        mne_raw.info['description'])	# Returns a 'Match' object

        if not (tmp is None):
            ninfo = tmp.group()
            if len(prefilter) > 0:
                prefilter = prefilter + ' '

            prefilter = prefilter + 'N:' + re.search(r"\d+\.*\d*", ninfo).group() + 'Hz'

    date = _stamp_to_dt(mne_raw.info['meas_date'])
    # no conversion necessary, as pyedflib can handle datetime.
    #date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None

    
    # convert data
    channels = mne_raw.get_data(picks, 
                                start = first_sample,
                                stop  = last_sample)
    
    # convert to microvolts to scale up precision
    # channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)

    # Dismiss warning regarding truncating trailing decimal digits when physical_min,
    # 'physical_max' are converted to a string longer than 8 characters. This happens
    # when those are found as channels.min(), channels.max() - see 'except:' processing
    # below
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message = 'Physical min')	# Warning from edfwriter.py`
        warnings.filterwarnings('ignore', message = 'Physical max')	# Warning from edfwriter.py`
        
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
                               'prefilter': prefilter}
                except:
                    ch_dict = {'label': mne_raw.ch_names[i], 
                               'dimension': mne_raw._orig_units[keys[i]], 
                               'sample_rate': sfreq, 
                               'physical_min': channels.min(), 
                               'physical_max': channels.max(), 
                               'digital_min':  dmin, 
                               'digital_max':  dmax, 
                               'transducer': '', 
                               'prefilter': prefilter}
            
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
