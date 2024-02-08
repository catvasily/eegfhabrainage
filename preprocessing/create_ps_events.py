import os.path as path
import re
import json
import mne

PREPROC_CONF_FILE = 'preproc_conf.json'
"""Name (without a path) for the JSON file with parameter settings for
the preprocessing step, which contains keywords that identify
photic stimulation intervals. This file is expected to reside in the same
folder as this script.
"""

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_PREPROC_CONF_PATHNAME = path.dirname(__file__) + "/" + PREPROC_CONF_FILE

def create_ps_events(raw, verbose=None):
    """
    Using `raw.annotations`, extract all events related to photic
    stimulation (PS) and create corresponding `events` array in MNE
    Python format.

    Args:
        raw (MNE Raw): the `Raw` object of the EEG record
        verbose (str or None): verbose level; one of ‘DEBUG’, ‘INFO',
            ERROR', 'CRITICAL', or 'WARNING' (default)

    Returns:
        events (ndarray): shape(nevents, 3) The MNE Python events array
            (https://mne.tools/1.4/glossary.html#term-events)
        event_id (dict): The `event_id` variable that can be passed
            to the `Epochs` object constructor (example: dict(auditory=1, visual=3)).
    """

    def ps_id(desc):    # External arguments: lst_starts, lst_ends
        """
        Given the description string of the event, return event ID
        for the photic stimulation (PS) event. Non-PS event are ignored.

        The integer IDs of the PS events are constructed as follows.
        For the start of the stimulation, corresponding event ID is 
        frequency in Hz, rounded to the nearest integer. For the end
        of stimulation with any frequency, the event ID is 0.

        Args:
            desc (str): event description, for example '10.0 Hz' or 'Off'
            lst_starts (list of str): (passed from the external host function)
                list of keywords identifying PS start event. Event is considered
                start of PS if any of the keywords is present in the description.
            lst_ends (list of str): (passed from the external host function)
                list of keywords identifying PS end event. Event is considered end
                of PS if any of the keywords is present in the description.

        Returns:
            id (int or None): the even ID, or None if non-PS event
        """
        a = desc.upper()
        for kword in lst_starts:
            if kword in a:
                return get_ps_frq(a)
                 
        for kword in lst_ends:
            if kword in a:
                return 0

        return None             

    # Load preprocessing config
    with open(_PREPROC_CONF_PATHNAME, "r") as fp:
        conf_dict = json.loads(fp.read())

    # Get the PS identification keywords
    lst_starts = [s.upper() for s in conf_dict["photic_starts"]]
    lst_ends = [s.upper() for s in conf_dict["photic_ends"]]

    return mne.events_from_annotations(raw,
            event_id=ps_id,
            regexp=None,
            use_rounding=True,
            chunk_duration=None,
            verbose=verbose)

def get_ps_frq(desc):
    """
    Read the PS frequency value from the PS event description
    and return it as an integer value. If not found, return
    `None`.
    """

    match = re.search(r'[0-9]*\.?[0-9]+', desc)

    if match:
        return int(round(float(match.group())))
    else:
        return None

if __name__ == '__main__':      # Unit tests 
    #raw_file = 'abbotsford_raw.fif'
    #raw_file = 'burnaby_raw.fif'
    #raw_file = 'rch_raw.fif'
    raw_file = 'surrey_raw.fif'
    raw = mne.io.read_raw_fif(raw_file, verbose = 'ERROR')

    events, ev_dict = create_ps_events(raw, verbose = 'ERROR')
    print(events)
    print(ev_dict)
