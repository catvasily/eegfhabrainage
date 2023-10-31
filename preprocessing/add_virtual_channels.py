'''
**A utility function to add virtual (source-reconstructed) channels to the raw dataset. This may be useful
for viewing/processing the EEG data using standard MNE Python routines.**
'''
import numpy as np
import mne
from mne.transforms import apply_trans, invert_transform
from mne.io.tag import _coil_trans_to_loc

def add_virtual_channels(raw, vc_names, vc_pos, vc_data, verbose = None):
    """Add virtual channels with data to an existing physical dataset.

    The virtual channels will have type matching the type of the physical
    sensors, their spatial locations will be set as specified, and their
    orientations will be set arbitrarily (and therefore should never be used).

    Args:
        raw (mne.Raw): the raw object with physical sensor channels.
        vc_names (list of str): a list of `n_vc` virtual channel names.
        vc_pos (ndarray): `n_vc x 3`; virtual sensors positions in **MNE HEAD**
            coordinate system, m
        vc_data (ndarray): `n_vc x n_times`; virtual channels time courses
        verbose (str): the verbose mode, or None for using MNE default mode.

    Returns:
        raw (mne.Raw): the modified in place instance of the input Raw object
    """

    n_vc, n_times = vc_data.shape

    if n_vc != len(vc_names):
        raise ValueError("length of vc_names should match vc_data.shape[0]")

    if n_times != raw.n_times:
        raise ValueError("vc_data.shape[1] should match raw.n_times")

    if vc_pos.shape != (n_vc, 3):
        raise ValueError("vc_pos.shape should must be equal to (n_vc, 3)")

    raw.load_data()	# Ensure that the data is allocated in memory 

    # Prepare a Raw instance with a single data channel
    raw1 = raw.copy().pick(['eeg','meg'], exclude = 'bads')
    drop_list = raw1.ch_names.copy()
    drop_list.pop(0)
    raw1.drop_channels(drop_list)
    current_name = raw1.ch_names[0]
    coil_trans = np.identity(4) 

    # Get virtual channels pos in dev coords
    # NOTE: for most EEG datasets dev_head_t is identity
    dev_head_t = raw.info['dev_head_t']	# the MNE dev -> head Transform object
    head_dev_t = invert_transform(dev_head_t)
    pos = apply_trans(head_dev_t, vc_pos, move=True)

    for i, vname in enumerate(vc_names):
        # Rename the channel to vname
        raw1.rename_channels({current_name: vname})
        current_name = vname

        # Get virtual channel pos in MNE device coordinates

        # Update the channel info
        # We leave the rotation part of transformation from "coil definition"
        # to device coords as identity transform, and only set the "offset" -
        # which is the sensor position in MNE device coordinates
        ch = raw1.info['chs'][0]
        coil_trans[:3,3] = pos[i]
        ch['loc'] = _coil_trans_to_loc(coil_trans)		# Generate corresponding 12D location vector
        # TODO: set the channel type and the units to something appropriate:
        # ch['unit'] = ...

        # Replace current data with the virtual channel data
        ch_data = vc_data[i,:]
        raw1._data = ch_data[np.newaxis,:]

        # Finally:
        raw.add_channels([raw1], force_update_info = True)
        # NOTE (AM): force_update_info will update info in raw1 to match that
        # in raw, except for keys["chs", "ch_names", "nchan"]. This does not matter
        # for us as raw1 is discarded anyways, but when setting flag to False
        # an exception is raised for some reason when adding a 2nd channel.

    return raw

# Unit test    
if __name__ == '__main__':
    fname = 'example_raw.fif' 
    vc_names = ['V1', 'V2']
    vc_pos = 1e-2 * np.array([[3, 3, 3], [-3, 3, 3]])
    verbose = 'INFO'
    seed = 12345

    raw = mne.io.read_raw(fname, verbose = verbose)

    rng = np.random.default_rng(seed = seed)
    vc_data = rng.random((len(vc_names), raw.n_times)) 

    add_virtual_channels(raw, vc_names, vc_pos, vc_data, verbose = None)
    print('New channel list:', raw.ch_names)

