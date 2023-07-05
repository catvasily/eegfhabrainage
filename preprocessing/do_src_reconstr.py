import sys
import os.path as path
import numpy as np
import h5py        # Needed to save/load files in .hdf5 format
import mne

sys.path.append(path.dirname(path.dirname(__file__))+ "/beam-python")

from nearest_pos_def import nearestPD
from construct_mcmv_weights import is_pos_def, construct_single_source_weights

def fwd_file_name(scan_id, src_file):
    """Construct forward solution file name base on the source space used 

    Args:
        scan_id (str): subject ID (=scan id)
        src_file (str): standard template source space name in the form
           fsaverage-....-src.fif

    Returns:
        fwd_name (str): fwd solutions file name in the form
            scan_id-....-fwd.fif
    """
    f = src_file.replace('fsaverage-', '{}-'.format(scan_id))
    f = f.replace('-src.fif', '-fwd.fif')
    return f

def get_voxel_coords(src, vertices):
    """Given vertex numbers, return voxel spatial coordinates for
    a surface source space.

    NOTE: the coordinate system is that of the source space; it may be
    either MRI or head coordinate system.

    Args:
        src (mne.SourceSpaces): as is
        vertices (list): a list `[lh, rh]`, where `lh, rh` are list of integer
           vertex numbers

    Returns:
        rr (ndarray): nvox x 3; coordinates of vertices. `nvox = len(lh) + len)rh)`
    """
    rr = np.append(
        src[0]['rr'][vertices[0],:],
        src[1]['rr'][vertices[1],:],
        axis = 0
        )

    return rr

def construct_noise_and_inv_cov(fwd, data_cov, *, tol = 1e-2, rcond = 1e-10):
    """Based on the forward solutions, construct sensor-level noise covariance
    matrix assuming white, randomly oriented uncorrelated sources. Also calculate
    pseudo-inverse of the data cov and SNR.

    The basic expression for noise covariance is:

    `cov0 = const * SUM(i=1 to Nsrc){Hx Hx' + Hy Hy' + Hz Hz'}`

    where `Hx,y,z(i)` are forward solutions for i-th source with corresponding
    orientations, and `const` is defined so as data_cov - noise_cov is non-negative.

    For degenerate data_cov the noise_cov should also be degenerate with a 
    range subspace coinciding with the range of the data_cov. In this case the
    above expression should be replaced with

    `cov = P * cov0 * P`

    where `P = data_cov * pinv(data_cov)` is a projector on the `range(data_cov)`.

    The trace of the noise_cov is maximized while keeping the difference
    `data_cov - noise_cov` non-negative. The tol parameter defines how close to
    the upper boundary one should get.

    Args:
        fwd (Forward): mne Python forward solutions class
        data_cov (ndarray): nchan x nchan data covariance matrix
        tol (float > 0): tolerance in finding the noise_cov trace.
        rcond (float > 0): singular values less than max(sing val) * rcond will
            be dropped

    Returns:
        cov (ndarray): (nchan x nchan) noise cov matrix, such that the difference
            data_cov - noise_cov is non-negatively defined
        inv_cov (ndarray): (nchan x nchan) (pseudo-) inverse of the data cov
        rank (int): rank of the data covariance
        pz (float): psedo-Z = SNR + 1 of the data; `pz=tr(data_cov)/tr(noise_cov)`
    """
    # Reduce all calcs to a full-rank subspace of the data_cov
    # U, Vh = nchan x nchan, S = (nchan,). In fact in this case Vh' = U
    U, S, Vh = np.linalg.svd(data_cov, full_matrices=False, hermitian=True)

    # Drop close to 0 singular values and corresponding columns of U. Note that all S[i]
    # are positive and sorted in decreasing order
    t = rcond * S[0]
    U = U[:, S > t]
    rank = U.shape[1]	# The actual rank of the data covariance

    H = fwd['sol']['data']	# Should be nchan x (3*nsrc) matrix

    # Reduced unnormalized noise covariance
    uH = U.T @ H
    unoise_cov = uH @ uH.T	# unoise_cov = U' H H' U

    # Reduced data covariance
    udata_cov = np.diag(S[:rank])    # udata_cov = U' U S U' U = S

    # (Pseudo-) inverse of the covariance
    # Make it pos def instead fully degenerate
    inv_cov = nearestPD(U @ np.diag(1./S[:rank]) @ U.T)

    # Initially, normalize it with the trace of data_cov
    pwr = np.trace(udata_cov)
    unoise_cov = (pwr / np.trace(unoise_cov)) * unoise_cov

    upper = pwr    # Current upper value of the trace (not pos def)
    lower = 0.     # Current lower value of the trace (already pos def)
    tr = pwr       # New value of the trace
    tr0 = tr       # Old value of the trace
 
    while True:
        if is_pos_def(udata_cov - unoise_cov):
            lower = tr
            tr = lower + (upper - lower)/2
            is_pd = True
        else:
            upper = tr
            tr = upper - (upper - lower)/2
            is_pd = False

        assert upper > lower				# Never happens, but just in case
        ratio = tr/tr0
        unoise_cov = ratio * unoise_cov
        tr0 = tr

        if is_pd and (np.abs(ratio - 1) < tol):
            break

    # Project results back to the original sensor space
    noise_cov = nearestPD(U @ unoise_cov @ U.T)
    pz = pwr / tr
        
    return noise_cov, inv_cov, rank, pz

def get_label_src_idx(fwd, label):
    """Get source indecies for a label (ROI)

    Forward solution matrices H and weight matrices W have
    columns (or triplets of columns) corresponding to sources
    in the whole (left + right hemisphere) source space. At the
    same time, the SourceSpaces object has separate source indexing
    for each hemisphere, and so does the ROI (label). This function
    returns a mapping from label vertices to columns of scalar H and
    scalar W.

    Args:
        fwd (mne.Forward): the global `Forward` object
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        idx (1D array of ints): n_label_src-dimensional vector of indecies
    """
    # Generally follow the SourceEstimate._hemilabel_stc() source code
    if label.hemi == 'lh':
        ihemi = 0
    elif label.hemi == 'rh':
        ihemi = 1
    else:
        raise ValueError("Only single hemisphere labels are allowed")

    # Get all the source space vertices for a hemisphere, that is
    # a mapping src # -> dense (FreeSurfer) vortex #
    all_vertices = fwd["src"][ihemi]["vertno"]    # 1D array of integers

    # Index of label vertices into all vertices of a hemisphere
    # Equivalently, idx yields source numbers belonging to the label
    idx = np.nonzero(np.in1d(all_vertices, label.vertices))[0]

    # Source space vertex numbers corresponding to the label, that is the
    # mapping src # -> dense (FreeSurfer) vortex # for a label (ROI)
    # label_vertices = all_vertices[idx]

    # In forward solutions or weights data, the left and right hemis are concatenated, so
    # source ## for the right hemisphere should be shifted by a total number of
    # sources in the left hemisphere:
    if ihemi == 1:
        idx += len(fwd["src"][0]["vertno"])

    return idx

def get_label_fwd(fwd, label):
    """Get a subset of forward solutions corresponding to specified Label (ROI)

    Args:
        fwd (mne.Forward): the global `Forward` object
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        H (ndarray): nchan x (3*n_label_src) array of the ROI forward solutions
    """
    idx = get_label_src_idx(fwd, label)

    # In matrix H there are 3 components for each source. Need to generate
    # a triple 3i, 3i+1, 3i+2 for each i
    l0 = 3*idx
    idx3 = (np.array([l0, l0 + 1, l0 + 2]).T).flatten()

    H = fwd['sol']['data']      # nchans x (3*nsrc)

    # Sanity check
    assert 3*(len(fwd["src"][0]["vertno"]) + len(fwd["src"][1]["vertno"])) == H.shape[1]

    return H[:,idx3]

def get_label_wts(fwd, W, label):
    """Get a subset of spatial filter weights corresponding to specified Label (ROI)

    Args:
        fwd (mne.Forward): the global `Forward` object
        W (ndarray): nchan x nsrc, weights matrix for the whole source space
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        W_label (ndarray): nchan x (n_label_src) array of ROI weights
    """
    assert fwd['sol']['data'].shape[0] == W.shape[0]
    assert int(fwd['sol']['data'].shape[1]/3) == W.shape[1]

    idx = get_label_src_idx(fwd, label)
    return W[:,idx]

def get_label_pca_weight(R, fwd, W, label):
    """Return a spatial filter vector `w_pca` such that a single label 
    (ROI) time course can be found by the expression `w_pca'*b(t)`, where
    `b` is a vector of sensor time courses. 

    *Explanation*. Covariance matrix of all signals that belong to a label is
    `R_label = W_label'* R * W_label`, where R is the global sensor covariance and
    `W_label = nchan x n_label_src` are label weights.
    Let `U0 = n_label_src x 1` be the largest normalized eigenvector of R_label.
    Then label time course `s(t)` corresponding to 'pca_flip' mode is found as

    `s(t) = sign * scale * U0' * W_label' * b(t)`,

    where

    `scale = sqrt[(trace(R_label)/E0)/n_label_src]`,
    `sign = np.sign(U0'*flip)`

    E0 is the largest eigenvalue of R_label and `flip` is a flip-vector returned by 
    MNE `label_sign_flip()` function. This scaling assigns the RMS of the powers
    of all ROI sources to the returned single time course amplitude. Then it is
    clear that the expression for `w_pca` is:

    `w_pca = sign * scale * W_label * U0, w_pca = nchan x 1`

    Args:
        R (ndarray): nchan x nchan, the global sensor data covariance matrix
        fwd (mne.Forward): the forward solutions object for the whole source space
        W (ndarray): nchan x nsrc, weights matrix for the whole source space
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        w_pca (ndarray): nchan x 1 weight vector for the ROI 
    """
    W_label = get_label_wts(fwd, W, label)
    R_label = W_label.T @ R @ W_label
    E, U = np.linalg.eigh(R_label)

    # The eigh() returns EVs in ascending order
    e0 = E[-1]
    U0 = U[:,-1]

    scale = np.sqrt(np.trace(R_label)/e0/W_label.shape[1])
    flip = mne.label_sign_flip(label, fwd['src'])
    sign = np.sign(U0.T @ flip)
    w_pca = sign * scale * (W_label @ U0)

    return w_pca

def get_beam_weights(H, inv_cov, noise_cov, units):
    """Get beamformer weights matrix for a set of forward solutions

    For each source, calculate a scalar beamformer weight using a formula

    `w = const * R^(-1) h; h = [Hx, Hy, Hz]*u`

    where `R` is the data covariance matrix, `h` is a "scalar" lead field corresponding
    to the source orientation `u`. The normalization constant is selected depending
    on the `units` parameter setting:

    `units = "source": const = (h' R^(-1) h)^(-1)`

    `units = "pz":     const = [h' R^(-1) N R^(-1) h]^(-1/2)`

    In the first case absolute current dipole amplitudes (A*m) will be reconstructed.
    In the 2nd case source amplitudes will be normalized on projected noise, effectively
    representing source-level signal to noise ratio.
 

    Args:
        H (ndarray): nchan x (3*n_src) array of FS for a set of n_src sources 
        inv_cov(ndarray): nchan x nchan (pseudo-)inverse of sesnor cov matrix
        noise_cov(ndarray): nchan x nchan noise cov matrix
        units (str): either "source" or "pz"

    Returns:
        W (ndarray): nchan x nsrc array of beamformer weights
        U (ndarray): 3 x nsrc array of source orientations
    """
    if units == "source":
        normalize = False
    elif units == "pz":
        normalize = True
    else:
        raise ValueError("The 'units' parameter value should be either 'source' or 'pz'")

    # Reshape forward solution matrix from nchan x (3*n_src) to n_src x n_chan x 3
    nchan, nsrc3 = H.shape
    nsrc = int(nsrc3 / 3)
    fs = np.reshape(H.T, (nsrc, 3, nchan))
    fs = np.transpose(fs, axes = (0, 2, 1))	# fs is nsrc x nchan x 3
    
    # Calculate beamformer weights; result corresponds to units = "source"
    # W is (nchan x nsrc), U is (3 x nsrc)
    W, U = construct_single_source_weights(fs, inv_cov, noise_cov, beam = "mpz", c_avg = None)

    if not normalize:
        return W

    # Normalize to get waveforms in pseudo-Z
    scales = np.sqrt(np.einsum('ns,nm,ms->s', W, noise_cov, W))	# A vector of nsrc values = sqrt(diag(W'N W))
    return W / scales, U

def compute_beamformer_stc(raw, fwd, *, return_stc = True, beam_type = 'pz', units = 'pz',
                           tol = 1e-2, rcond = 1e-10, verbose = None):
    """ Reconstruct source time courses using single source minimum variance
    beamformer.

    Args:
        raw (mne.Raw): the raw data containing only good sensor channels
        fwd (mne.Forward): forward solutions
        return_stc (bool): flag to compute all source time courses and to return
            corresponding SourceEstimate object
        beam_type (str): beamformer type: one of `'pz'` (pseudo-Z) or `'ai'` (activity
            index) scalar beamformer can be calculated.
        tol (float > 0): tolerance (relative accuracy) in finding the noise_cov trace.
        rcond (float > 0): condition for determining rank of the covariance matrix:
            singular values less than `max(sing val) * rcond` will be considered zero.
        verbose (str or None): verbose mode (see MNE docs for details)

    Returns:
        stc (mne.SourceEstimate or None): source estimate (reconstructed source time courses) for
            a surfaced based source space, provided return_stc flag is True; None otherwise
        data_cov (ndarray): nchan x nchan, sensor covariance matrix adjusted to a nearest
            positive definite matrix
        W (ndarray): nchan x nsrc array of beamformer weights
        U (ndarray): 3 x nsrc array of source orientations
    """
    # Compute sample covariance
    eeg_data = raw.get_data(    # eeg_data is nchannels x ntimes
        picks = 'eeg',          # bads are already dropped
        start=0,                # starting time sample number (int)
        stop=None,
        reject_by_annotation=None,
        return_times=False,
        units=None,             # return SI units
        verbose=verbose)

    nchan = eeg_data.shape[0]

    # We use nearestPD() because degenerate cov may have negative EVs
    # due to rounding errors
    data_cov = nearestPD(np.cov(eeg_data, rowvar=True, bias=False))

    # Data covariance is always degenerate due to avg ref, interpolated channels, etc.
    noise_cov, inv_cov, rank, pz = construct_noise_and_inv_cov(fwd, data_cov, tol = tol, rcond = rcond)

    if verbose == 'INFO':
        print('Data covarince: nchan = {}, rank = {}'.format(nchan, rank))

    # W is nchan x nsrc, U is 3 x nsrc
    W, U = get_beam_weights(fwd['sol']['data'], inv_cov, noise_cov, units = units) 

    # Flip orientations which are more than 180 degrees off the sourface orientations
    assert fwd['src'][0]['type'] == 'surf'
    assert fwd['src'][0]['coord_frame'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD

    Usurf = list()
    vertices = list()

    for ihemi in (0, 1):
        src = fwd['src'][ihemi]
        vts = src['vertno']
        Usurf.append(src['nn'][vts,:])
        vertices.append(vts)

    Usurf = np.concatenate(Usurf, axis = 0)	# This is nsrc x 3 array
    assert Usurf.shape[0] == U.shape[1]

    signs = np.sign(np.einsum('ij,ji->i', Usurf, U))	# A vector of nsrc 1s, -1s
    W *= signs    # It does flip the columns of W, U due to np broadcast rules 
    U *= signs 

    # We normalize time sources on sqrt(PZ), which is equivalent to have tr(N) = tr(R).
    # This way time courses for all subjects will be scaled identically. Proof:
    #    pz = tr(R) / tr(N) by definition
    #    w = Rinv*h/sqrt(hRinv N Rinv h) = sqrt(pz) w1,
    #  where
    #    w1 = Rinv*h/sqrt[hRinv (pz*N) Rinv h]
    # Obviously tr(pz*N) = tr(R), thus w1 corresponds to noise with power equal to that of R.
    # So by normalizing w on sqrt(pz) we ensure tr(R) = tr(N) for all subjects.
    W /= np.sqrt(pz)

    # Create SourceEstimate object to return
    if return_stc:
        src_data = (W.T @ eeg_data)
        stc = mne.SourceEstimate(
            src_data,
            vertices, 
            tmin = raw.times[0],
            tstep = raw.times[1] - raw.times[0],
            subject='fsaverage',
            verbose=verbose)
    else:
        stc = None

    return stc, data_cov, W, U

def compute_source_timecourses(raw, fwd, *, method = "beam", return_stc = True, **kwargs): 
    """ Reconstruct source time courses for all sources in the source space

    Args:
        raw (mne.Raw): the raw data containing only good sensor channels
        fwd (mne.Forward): forward solutions
        method (str): source reconstruction method; currently only beamformer
            (method = 'beam') is implemented
        return_stc (bool): flag to compute all source time courses and to return
            corresponding SourceEstimate object
        kwargs (dict): dictionary with method-specific arguments

    Returns:
        stc (mne.SourceEstimate or None): source estimate (reconstructed source time courses) for
            a surfaced based source space, if `return_stc = True`, or None otherwise.
        data_cov (ndarray or None): nchan x nchan; if `method = 'beam'`: sensor covariance matrix
            adjusted to a nearest positive definite matrix; None otherwise
        W (ndarray or None): nchan x nsrc; if `method = 'beam'`:  array of beamformer weights;
            None othewise
        U (ndarray or None): 3 x nsrc; if `method = 'beam'`: array of source orientations;
            None otherwise
    """

    if method == 'beam':
        return compute_beamformer_stc(raw, fwd, return_stc = return_stc, **kwargs) 

    raise ValueError('Method {} is unknown or not implemented'.format(method))

def ltc_file_name(scan_id, src_file):
    """Construct HDF5 file name for saved ROI time courses based on the source space used 

    Note that the names of ROIs themselves depend on the atlas (parcellation) used and
    will be stored inside the generated HDF5 file.

    Args:
        scan_id (str): subject ID (=scan id)
        src_file (str): standard template source space name in the form
           fsaverage-....-src.fif

    Returns:
        fwd_name (str): fwd solutions file name in the form
            scan_id-....-ltc.hdf5
    """
    f = src_file.replace('fsaverage-', '{}-'.format(scan_id))
    f = f.replace('-src.fif', '-ltc.hdf5')
    return f 

def read_roi_time_courses(ltc_file):
    """Save ROI (label) time courses and corresponding ROI names in .hdf5
    file.

    Args:
        ltc_file (str): full pathname of the output .hdf5 file

    Returns:
        label_tcs (ndarray): nlabels x ntimes ROI time courses
        label_names (ndarray of str):  1 x nlabels vector of ROI names
        rr (ndarray or None): nlabels x 3; coordinates of ROI reference locations
            in head coordinates
        W (ndarray or None): nchans x nlabels; spatial filter weights for each ROI.
            Those can be used to reconstruct ROI time courses as `W.T @ sensor_data` 
    """
    with h5py.File(ltc_file, 'r') as f:
        label_tcs = f['label_tcs'][:,:]
        label_names = f['label_names'].asstr()[:]

        if 'rr' in f:
            rr = f['rr'][:,:]
        else:
            rr = None

        if 'W' in f:
            W = f['W'][:,:]
        else:
            W = None

    return (label_tcs, label_names, rr, W)  
 
def write_roi_time_courses(ltc_file, label_tcs, label_names, rr = None, W = None):
    """Save ROI (label) time courses and corresponding ROI names in .hdf5
    file.

    The output file will contain at least two datasets with names 'label_tcs' and
    'label_names'. If provided, ROI coordinates and corresponding spatial filter
    weights will also be saved under names 'rr' and 'W', respectively.

    Args:
        ltc_file (str): full pathname of the output .hdf5 file
        label_tcs (ndarray): nlabels x ntimes ROI time courses
        label_names (list of str): names of ROIs
        rr (ndarray or None): nlabels x 3; coordinates of ROI reference locations
            in head coordinates
        W (ndarray or None): nchans x nlabels; spatial filter weights for each ROI.
            Those can be used to reconstruct ROI time courses as `W.T @ sensor_data` 

    Returns:
        None
    """
    with h5py.File(ltc_file, 'w') as f:
        f.create_dataset('label_tcs', data=label_tcs)
        f.create_dataset('label_names', data=label_names)

        if not (rr is None):
            f.create_dataset('rr', data=rr)

        if not (W is None):
            f.create_dataset('W', data=W)

def beam_extract_label_time_course(sensor_data, cov, labels, fwd, W, mode = 'pca_flip',
        verbose = None):
    """Compute spatial filter weights and time courses for ROIs (labels) using beamformer
    inverse solutions.

    Args:
        sensor_data (ndarray): nchan x ntimes; EEG channels time courses
        cov (ndarray): nchan x nchan; the sensor time courses covariance matrix
        labels (list): a list of mne.Label objects for the ROIs 
        fwd (mne.Forward): forward solutions
        W (ndarray): nchan x nsrc; beamformer weights for the whole (global) source space
        mode (str): a method of constructing a single time course for the ROI; see description
            of `mne.extract_label_time_course()` function
        verbose (str): verbose mode

    Returns:
        label_tcs (ndarray): nlabels x ntimes; ROI time courses
        label_wts (ndarray): nchan x nlabels; spatial filter weights for each label
    """
    roi_modes = {'pca_flip': get_label_pca_weight}
    
    if not mode in roi_modes:
        raise ValueError('Mode {} is unknown or not supported'.format(mode))

    if verbose == 'INFO':
        print('Reconstructing ROI time courses using beamformer weights, mode = {}'.format(mode))

    func = roi_modes[mode]
    nlabels = len(labels)
    nchans, ntimes = sensor_data.shape
    label_wts = np.zeros((nchans, nlabels))

    for i,label in enumerate(labels):
        label_wts[:, i] = func(cov, fwd, W, label)

    label_tcs = label_wts.T @ sensor_data

    return label_tcs, label_wts
        
def compute_roi_time_courses(inv_method, labels, fwd, mode = 'pca_flip',
        stc = None, sensor_data = None, cov = None, W = None, verbose = None):
    """Compute time courses for ROIs (labels).

    If `inv_method` is 'beam' and `mode` is `pca_flip` - `beam_extract_label_time_course()`
    will be used; otherwise a MNE function  mne.extract_label_time_course() will be applied.

    Args:
        inv_method (str): inverse method used to create the `stc`
        labels (list): a list of mne.Label objects for the ROIs 
        fwd (mne.Forward): forward solutions
        mode (str): a method of constructing a single time course for ROI - see description
            of `mne.extract_label_time_course() function.
        stc (mne.SourceEstimate or None): source estimate (reconstructed source time courses) for
            a surfaced based source space; only needed for non-beamformer reconstructions
        sensor_data (ndarray or None): nchan x ntimes; EEG channels time courses. Must be
            specified for beamformer reconstructions, not needed for others.
        cov (ndarray or None): nchan x nchan; the sensor time courses covariance matrix. Must be
            specified for beamformer reconstructions, not needed for others.
        W (ndarray or None): nchan x nsrc; beamformer weights for the whole (global) source space.
             Must be specified for beamformer reconstructions, not needed for others.
        verbose (str): verbose mode

    Returns:
        label_tcs (ndarray): nlabels x ntimes; ROI time courses
        label_wts (ndarray or None): nchan x nlabels; spatial filter weights for each label for
            beamformer reconstructions, None for other (min norm) reconstructions.
    """
    beam_modes = ['pca_flip']    # A list of modes supported by beam_extract_label_time_course() 

    if (inv_method == "beam") and (mode in beam_modes):
        label_tcs, label_wts = beam_extract_label_time_course(sensor_data, cov, labels,
                                   fwd, W, mode = mode, verbose = verbose)
    else: 
        label_tcs = mne.extract_label_time_course(stc, labels, fwd['src'],
            mode=mode,                 # How to extract a time course for ROI
            allow_empty=False,         # Raise exception for empty ROI 
            return_generator=False,    # Return nRoi x nTimes matrix, not a generator
            mri_resolution=False,      # Do not upsample source space
            verbose=verbose)

        label_wts = None

    return label_tcs, label_wts

