"""
**Build \"epi_features\" dimensionality reducer.**
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def build_and_fit_epi_features(raw_features, config_dict, standardize=False):
    """
    Build and fit the 'epi_features' dimensionality reducer.

    Args:
        raw_features(ndarray): shape (nscans, nchans, nfreqs, nparms)
        config_dict(dict): reducer config and metadata. Must include:
            - freqs: list/array of frequency values in Hz
            - parm_names: list of parameter names including 'mean', 'skew', 'kurtosis'
            Optional:
            - epi_features.bands: dict with keys delta/theta/alpha/beta, values [low, high]
            - epi_features.ratio_eps: small epsilon for safe division
        standardize(bool): whether to standardize final flattened features

    Returns:
        X(ndarray): shape (nscans, nfeatures)
        transformer(dict): fitted reducer payload compatible with apply_fitted_reducer()

    """
    freqs = np.asarray(config_dict.get('freqs', []), dtype=float)
    parm_names = list(config_dict.get('parm_names', []))

    if freqs.size == 0:
        raise ValueError('epi_features reducer requires frequency vector in config_dict["freqs"]')

    required_parms = ['mean', 'skew', 'kurtosis']
    missing = [name for name in required_parms if name not in parm_names]

    if missing:
        raise ValueError(f'epi_features reducer missing required parm(s): {missing}')

    epi_cfg = config_dict.get('epi_features', {}) or {}
    band_cfg = epi_cfg.get('bands', {}) or {}

    # Defaults are clinical EEG bands from user requirements.
    bands = {
        'delta': tuple(band_cfg.get('delta', [0.5, 4.0])),
        'theta': tuple(band_cfg.get('theta', [4.0, 7.0])),
        'alpha': tuple(band_cfg.get('alpha', [7.0, 14.0])),
        'beta': tuple(band_cfg.get('beta', [14.0, 30.0])),
    }
    ratio_eps = float(epi_cfg.get('ratio_eps', 1e-12))

    # Precompute masks from frequency values (works for non-linear/log spacing).
    band_masks = {name: _freq_mask(freqs, bounds, name) for name, bounds in bands.items()}

    parm_indices = {
        'mean': parm_names.index('mean'),
        'skew': parm_names.index('skew'),
        'kurtosis': parm_names.index('kurtosis'),
    }

    features_3d = _compute_epi_features(raw_features, freqs, parm_indices, band_masks, ratio_eps)
    ch_names = list(config_dict.get('ch_names', []))
    if ch_names:
        if len(ch_names) != features_3d.shape[1]:
            raise ValueError(
                'epi_features reducer channel mismatch: '
                f'config_dict["ch_names"] has {len(ch_names)} channels, '
                f'features have {features_3d.shape[1]}'
            )
        channel_ops, paired_hemi_bases, unpaired_hemi_ch_names = _build_hemi_channel_ops(ch_names)
        features_3d = _apply_channel_ops(features_3d, channel_ops)
        ch_names_out = [spec['name'] for spec in channel_ops]
    else:
        channel_ops = [{'op': 'copy', 'name': f'c{i}', 'src_idx': int(i)} for i in range(features_3d.shape[1])]
        ch_names_out = [spec['name'] for spec in channel_ops]
        paired_hemi_bases = []
        unpaired_hemi_ch_names = []

    nscans = features_3d.shape[0]
    features_flat = features_3d.reshape(nscans, -1)

    if standardize:
        pipeline = make_pipeline(StandardScaler())
        X = pipeline.fit_transform(features_flat)
        print('  Standardization applied')
    else:
        pipeline = None
        X = features_flat

    epi_parm_names = ['r_dt', 'r_ta', 'r_ab']
    for band_name in band_masks:
        epi_parm_names.extend([
            f'{band_name}_mean_skew',
            f'{band_name}_std_skew',
            f'{band_name}_mean_kurt',
            f'{band_name}_std_kurt',
            f'{band_name}_f_kmax',
        ])

    transformer = {
        'method': 'epi_features',
        'pipeline': pipeline,
        'epi_parm_names': epi_parm_names,
        'bands': bands,
        'ratio_eps': ratio_eps,
        'freqs': freqs,
        'band_masks': band_masks,
        'parm_indices': parm_indices,
        'channel_ops': channel_ops,
        'ch_names_out': ch_names_out,
        'paired_hemi_bases': paired_hemi_bases,
        'unpaired_hemi_ch_names': unpaired_hemi_ch_names,
    }

    print(f'  Reduced to epi_features with {len(epi_parm_names)} parameters per channel')
    return X, transformer


def apply_epi_features(raw_features, transformer):
    """
    Apply a fitted 'epi_features' reducer to raw features.

    Args:
        raw_features(ndarray): shape (nscans, nchans, nfreqs, nparms)
        transformer(dict): reducer dict returned by build_and_fit_epi_features()

    Returns:
        X(ndarray): shape (nscans, nfeatures)
    """
    freqs = np.asarray(transformer['freqs'], dtype=float)
    parm_indices = transformer['parm_indices']
    band_masks = transformer['band_masks']
    ratio_eps = float(transformer.get('ratio_eps', 1e-12))

    if raw_features.shape[2] != freqs.size:
        raise ValueError(
            'epi_features transformer frequency dimension mismatch: '
            f'raw_features has nfreqs={raw_features.shape[2]}, transformer expects {freqs.size}'
        )

    features_3d = _compute_epi_features(raw_features, freqs, parm_indices, band_masks, ratio_eps)
    channel_ops = transformer.get('channel_ops')
    if channel_ops is None:
        channel_ops = [{'op': 'copy', 'name': f'c{i}', 'src_idx': int(i)} for i in range(features_3d.shape[1])]
    features_3d = _apply_channel_ops(features_3d, channel_ops)

    nscans = features_3d.shape[0]
    features_flat = features_3d.reshape(nscans, -1)

    pipeline = transformer.get('pipeline')
    if pipeline is None:
        return features_flat

    return pipeline.transform(features_flat)


def _freq_mask(freqs, bounds, name):
    """
    Return idx mask for freqs that belong to the band.
    `name` parm is only needed for the error message
    """
    low, high = [float(v) for v in bounds]
    if low > high:
        raise ValueError(f'{name} bounds must satisfy low <= high, got {bounds}')

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        raise ValueError(f'{name}={bounds} does not overlap frequency grid [{freqs.min()}, {freqs.max()}] Hz')
    return mask


def _safe_ratio(num, den, eps):
    return num / (den + eps)


def _compute_epi_features(raw_features, freqs, parm_indices, band_masks, ratio_eps):
    """
    Extract mean/skew/kurtosis planes and compute per-channel epi features.

    Features include three global power ratios (`r_dt`, `r_ta`, `r_ab`) plus,
    for each band in ``band_masks`` order, skew/kurtosis statistics and the
    kurtosis peak frequency inside that band.

    Args:
        raw_features (np.ndarray): Original full feature array shaped
            ``(n_scans, n_channels, n_freqs, n_params)``.
        freqs (np.ndarray): Frequency grid shaped ``(n_freqs,)``.
        parm_indices (dict[str, int]): Mapping with keys ``'mean'``, ``'skew'``,
            and ``'kurtosis'`` pointing to parameter indices in ``raw_features``.
        band_masks (dict[str, np.ndarray]): Boolean masks over ``freqs`` for the
            canonical bands (typically ``'delta'``, ``'theta'``, ``'alpha'``,
            ``'beta'``). The same masks are used for per-band skew/kurtosis
            summary features.
        ratio_eps (float): Small denominator stabilizer used in ratio features.

    Returns:
        np.ndarray: Epi features shaped
            ``(n_scans, n_channels, 3 + 5 * n_bands)`` with parms ordered as
            ``[r_dt, r_ta, r_ab,`` then per-band
            ``mean_skew, std_skew, mean_kurt, std_kurt, f_kmax]``.

    """
    mean_values = raw_features[:, :, :, parm_indices['mean']]       # shape (nscans, nchan, nfreq)
    skew_values = raw_features[:, :, :, parm_indices['skew']]       # --"--
    kurt_values = raw_features[:, :, :, parm_indices['kurtosis']]   # --"--

    # Calculate powers across specified bands as pwr = sum(band){mean^2}
    p_delta = np.sum(np.square(mean_values[:, :, band_masks['delta']]), axis=2)
    p_theta = np.sum(np.square(mean_values[:, :, band_masks['theta']]), axis=2)
    p_alpha = np.sum(np.square(mean_values[:, :, band_masks['alpha']]), axis=2)
    p_beta = np.sum(np.square(mean_values[:, :, band_masks['beta']]), axis=2)

    # Calculate band power ratios
    r_dt = _safe_ratio(p_delta, p_theta, ratio_eps)
    r_ta = _safe_ratio(p_theta, p_alpha, ratio_eps)
    r_ab = _safe_ratio(p_alpha, p_beta, ratio_eps)

    per_band_features = []
    for band_name, band_mask in band_masks.items():
        band_skew = skew_values[:, :, band_mask]
        band_kurt = kurt_values[:, :, band_mask]

        mean_skew = np.mean(band_skew, axis=2)
        std_skew = np.std(band_skew, axis=2)
        mean_kurt = np.mean(band_kurt, axis=2)
        std_kurt = np.std(band_kurt, axis=2)

        band_freqs = freqs[band_mask]
        kmax_idx = np.argmax(band_kurt, axis=2)
        f_kmax = band_freqs[kmax_idx]

        per_band_features.extend([mean_skew, std_skew, mean_kurt, std_kurt, f_kmax])

    features_3d = np.stack([r_dt, r_ta, r_ab, *per_band_features], axis=2)
    return features_3d

def _build_hemi_channel_ops(ch_names):
    """
    Build channel transform ops from -lh/-rh suffix pairs.

    For each paired base name, emit two channels:
      - <base>-avg = 0.5 * (lh + rh)
      - <base>-dif = rh - lh
    Channels without a pair are copied unchanged.

    Returns:
        ops(list of dict): a list of dictionaries with keys
            'op' ('avg,'dif' or 'copy'); 'name' (ch base name); then either
            'lh_idx','rh_idx' = idx of ch in original list, or 'src_idx' for
            unpaired channel;
        paired_bases(list of str): list of basenames of paired channels
        unpaired_hemi(list of str): list of chnames found only in 1 hemi

    """
    lh_by_base = {}
    rh_by_base = {}
    for idx, name in enumerate(ch_names):
        if name.endswith('-lh'):
            lh_by_base[name[:-3]] = idx
        elif name.endswith('-rh'):
            rh_by_base[name[:-3]] = idx

    paired_bases = []   # basenames of paired channels
    used_pair_bases = set()

    # Collect paired channels
    for name in ch_names:
        if name.endswith('-lh') or name.endswith('-rh'):
            base = name[:-3]
            if base in used_pair_bases:
                continue
            if (base in lh_by_base) and (base in rh_by_base):
                paired_bases.append(base)
                used_pair_bases.add(base)

    paired_base_set = set(paired_bases)
    unpaired_hemi = []      # chnames of unpaired channels
    copy_ops = []

    # Collect unpaired channels and create 'ops' sublist for those,
    # with the 'op' set to 'copy'
    for idx, name in enumerate(ch_names):
        if name.endswith('-lh') or name.endswith('-rh'):
            base = name[:-3]

            if base in paired_base_set:
                continue

            copy_ops.append({'op': 'copy', 'name': name, 'src_idx': int(idx)})
            unpaired_hemi.append(name)
        else:
            copy_ops.append({'op': 'copy', 'name': name, 'src_idx': int(idx)})

    # Compile 'ops' sublist with 'avg' op:
    avg_ops = [
        {'op': 'avg', 'name': f'{base}-avg', 'lh_idx': int(lh_by_base[base]), 'rh_idx': int(rh_by_base[base])}
        for base in paired_bases
    ]

    # Compile 'ops' sublist with 'dif' op:
    dif_ops = [
        {'op': 'dif', 'name': f'{base}-dif', 'lh_idx': int(lh_by_base[base]), 'rh_idx': int(rh_by_base[base])}
        for base in paired_bases
    ]

    # Construct full ops list in the order: 'avg' chans, 'dif' chans, unpaired chans
    ops = avg_ops + dif_ops + copy_ops

    return ops, paired_bases, unpaired_hemi

def _apply_channel_ops(features_3d, channel_ops):
    """
    Apply channel ops returned by `_build_hemi_channel_ops()`
    to `features_3d (nscans, nchans, nparms)`.

    Returns:
        new_features_3d(nparray): `(nscans,nchans,nparms)` where the channels
            order is avg channels, dif channels, unpaired channels

    """
    transformed = []    # A list of 2D arrays with shape (nscans, nparms)

    # NOTE that the order of channels in the output is defined by the order of
    # specs in the channel_ops - which is avg, dif, unpaired
    for spec in channel_ops:
        op = spec['op']
        if op == 'copy':
            transformed.append(features_3d[:, spec['src_idx'], :])
        elif op == 'avg':
            lh_idx = spec['lh_idx']
            rh_idx = spec['rh_idx']
            transformed.append(0.5 * (features_3d[:, lh_idx, :] + features_3d[:, rh_idx, :]))
        elif op == 'dif':
            lh_idx = spec['lh_idx']
            rh_idx = spec['rh_idx']
            transformed.append(features_3d[:, rh_idx, :] - features_3d[:, lh_idx, :])
        else:
            raise ValueError(f'Unknown channel op: {op}')

    return np.stack(transformed, axis=1)


