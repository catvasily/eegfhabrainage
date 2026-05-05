"""
**Show importances of features identified by the classifier.**
"""
import numpy as np
from pathlib import Path
import pickle
import warnings
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from do_xgboost import verify_pickle_xgboost_version, load_xgb_model_from_ubj_buffer
from plot_3d_heatmap import (
    plot_3d_heatmap,
    _format_frequency_labels,
    _major_tick_positions,
    _round_frequency_ticks,
    _resolve_y_limits,
)

STEP = 'feature_importance'
_ALLOWED_IMPORTANCE_TYPES = {'gain', 'weight', 'cover'}

def cls_feature_importance(ss):
    """
    For specified target label, load trained model from corresponding
    pickle file and plot feature importances.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    # Read top-level feature-importance config and method-specific subkeys.
    fi_cfg = ss.args.get('feature_importance')

    if fi_cfg is None:
        fi_cfg = ss.args.get('feature_importnace')

    fi_cfg = fi_cfg or {}

    # Load the saved classifier payload and reconstruct the trained XGBoost model.
    pkl_pname = _find_results_pickle(ss)
    print(f'Loading classification results from: {pkl_pname}')

    with open(pkl_pname, 'rb') as fp:
        # Supress warning that pickle could be created with different
        # version of XGBoost
        with warnings.catch_warnings():          
            warnings.filterwarnings(            
                'ignore',
                message=r'.*If you are loading a serialized model.*',
                category=Warning,
            )
            payload = pickle.load(fp)

    verify_pickle_xgboost_version(payload, pkl_pname)

    full_model_data = payload.get('full_model_data') if isinstance(payload, dict) else None

    if not isinstance(full_model_data, dict):
        raise KeyError('Missing "full_model_data" in classification pickle payload')

    model_ubj = full_model_data.get('model_ubj')
    reducer = full_model_data.get('reducer')
    feature_names = full_model_data.get('feature_names')

    # Currently, the following formats for feature names are used:
    #   - to_lobes: '<L or R>-<lobe>>|X.ddHz|<stat-moment>', for example: 'R-I|55.00Hz|kurtosis'
    #   - CCV:      '<L or R>-<lobe>|ch<N>|X.ddHz|<stat-moment>', for example: 'R-T|ch146|55.00Hz|std'

    model = load_xgb_model_from_ubj_buffer(model_ubj)

    if model is None:
        raise KeyError('Missing trained model reconstructed from full_model_data["model_ubj"]')

    if not isinstance(reducer, dict):
        raise KeyError('Missing reducer metadata in full_model_data["reducer"]')

    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise KeyError('Missing or empty full_model_data["feature_names"]')

    method = reducer.get('method', '')
    input_stem_suffix = _extract_ijob_suffix(pkl_pname) # Return string '_ijob<N>' if present

    if method == 'to_lobes':
        to_lobes_cfg = fi_cfg.get('to_lobes', {})
        to_lobes_cfg = to_lobes_cfg if isinstance(to_lobes_cfg, dict) else {}

        # _run_feature_importance_to_lobes() is specific to the case when ROIs
        # in the reduced feature set are not present, and lobes (groups) are
        # used instead
        lobes_heatmap_cfg = fi_cfg.get('lobes_heatmap', {})
        lobes_heatmap_cfg = lobes_heatmap_cfg if isinstance(lobes_heatmap_cfg, dict) else {}

        _run_feature_importance_to_lobes(
            ss,
            payload,
            reducer,
            feature_names,
            model,
            to_lobes_cfg,
            lobes_heatmap_cfg,
            input_stem_suffix=input_stem_suffix,
        )

    elif method == 'consensus_cv':
        # _run_feature_importance_ccv() is NOT specific to the CCV case as such;
        # rather it is assumed that features from the reduced set can still be
        # represented as (ROI, frequency, parm) triple.
        ccv_cfg = fi_cfg.get('ccv', {})
        ccv_cfg = ccv_cfg if isinstance(ccv_cfg, dict) else {}
        _run_feature_importance_ccv(
            ss,
            payload,
            reducer,
            feature_names,
            model,
            ccv_cfg,
            input_stem_suffix=input_stem_suffix,
        )

    else:
        raise ValueError(
            f'feature_importance currently supports dim_reduction in '
            f'{{"to_lobes", "consensus_cv"}} only; found "{method}"'
        )

    print(f'\n Step *{STEP}* completed successfully')


def _run_feature_importance_to_lobes(
    ss,
    payload,
    reducer,
    feature_names,
    model,
    to_lobes_cfg,
    lobes_heatmap_cfg,
    input_stem_suffix='',
):
    """
    Run feature-importance post-processing and plots for `to_lobes` reduction.

    Args:
        ss (obj): Application object containing runtime settings and paths.
        payload (dict): Classification payload loaded from pickle.
        reducer (dict): Dimensionality-reduction metadata.
        feature_names (Sequence[str]): Flattened model feature names.
        model (xgboost.XGBModel): Trained model reconstructed from payload.
        to_lobes_cfg (dict): Method-specific config under
            ``feature_importance.to_lobes``.
        lobes_heatmap_cfg (dict): Optional plotting config under
            ``feature_importance.lobes_heatmap`` used when
            ``feature_importance.to_lobes.use_lobes_heatmap`` is true.
        input_stem_suffix (str): Optional stem suffix propagated from the input
            classifier pickle name (for example ``_ijob7``), appended to output
            plot file stems.

    Returns:
        None

    """
    importance_type = str(to_lobes_cfg.get('ranking', ss.args.get('feature_ranking', 'gain'))).lower()

    if importance_type not in _ALLOWED_IMPORTANCE_TYPES:
        raise ValueError(
            f'Unsupported feature_importance.to_lobes.ranking="{importance_type}". '
            f'Allowed values are: {sorted(_ALLOWED_IMPORTANCE_TYPES)}'
        )

    gain_axis_scale = str(to_lobes_cfg.get('gain_axis_scale', ss.args.get('gain_axis_scale', 'log'))).lower()

    if gain_axis_scale not in ('linear', 'log'):
        raise ValueError(
            'Unsupported feature_importance.to_lobes.gain_axis_scale. '
            'Allowed values are: ["linear", "log"]'
        )

    font_scale = float(to_lobes_cfg.get('font_scale', ss.args.get('feature_importance_font_scale', 0.75)))

    if font_scale <= 0:
        raise ValueError('feature_importance.to_lobes.font_scale must be > 0')

    use_lobes_heatmap = bool(to_lobes_cfg.get('use_lobes_heatmap', False))

    figure_size_cfg = to_lobes_cfg.get('figure_size', None)
    y_limits_cfg = to_lobes_cfg.get('y_limits', None)
    group_summary_figure_size_cfg = to_lobes_cfg.get('lobe_summary_figure_size', None)
    freq_summary_figure_size_cfg = to_lobes_cfg.get('frequency_summary_figure_size', None)
    parm_summary_figure_size_cfg = to_lobes_cfg.get('parameter_summary_figure_size', None)
    group_summary_y_limits_cfg = to_lobes_cfg.get('lobe_summary_y_limits', None)
    freq_summary_y_limits_cfg = to_lobes_cfg.get('frequency_summary_y_limits', None)
    parm_summary_y_limits_cfg = to_lobes_cfg.get('parameter_summary_y_limits', None)

    group_labels = reducer.get('group_labels', [])
    ngroups, nfreqs, nparms, freq_labels, parm_labels = _infer_to_lobes_dims(feature_names, group_labels)

    feature_importance_1d = _extract_importance_vector(model, feature_names, importance_type)

    if feature_importance_1d.shape[0] != len(feature_names):
        raise RuntimeError('Feature importance length mismatch with feature_names')

    feature_importance_3d = feature_importance_1d.reshape(ngroups, nfreqs, nparms)
    feature_names_3d = np.asarray(feature_names, dtype=object).reshape(ngroups, nfreqs, nparms)

    ss.feature_importance_1d = feature_importance_1d
    ss.feature_importance_3d = feature_importance_3d
    ss.feature_names_3d = feature_names_3d
    ss.feature_importance_meta = {
        'importance_type': importance_type,
        'group_labels': group_labels,
        'freq_labels': freq_labels,
        'parm_labels': parm_labels,
    }

    label = str(payload.get('target_label', ss.args.get('target_label', 'label')))
    ignore_confidence = bool(ss.args.get('ignore_confidence', False))
    outfname = Path(ss.out_root) / (
        f'feature_importance_{label}_{importance_type}_ignoreConf{ignore_confidence}'
        f'{input_stem_suffix}.png'
    )
    outfname_group = Path(ss.out_root) / (
        f'feature_importance_group_summary_{label}_{importance_type}_ignoreConf{ignore_confidence}'
        f'{input_stem_suffix}.png'
    )
    outfname_freq = Path(ss.out_root) / (
        f'feature_importance_frequency_summary_{label}_{importance_type}_ignoreConf{ignore_confidence}'
        f'{input_stem_suffix}.png'
    )
    outfname_parm = Path(ss.out_root) / (
        f'feature_importance_parameter_summary_{label}_{importance_type}_ignoreConf{ignore_confidence}'
        f'{input_stem_suffix}.png'
    )
    outfname_lobes_heatmap = Path(ss.out_root) / (
        f'feature_importance_lobes_heatmap_{label}_{importance_type}_ignoreConf{ignore_confidence}'
        f'{input_stem_suffix}.png'
    )

    if use_lobes_heatmap:
        _run_to_lobes_heatmap_branch(
            ss=ss,
            feature_importance_3d=feature_importance_3d,
            feature_names_3d=feature_names_3d,
            group_labels=group_labels,
            freq_labels=freq_labels,
            parm_labels=parm_labels,
            lobes_heatmap_cfg=lobes_heatmap_cfg,
            outfname_lobes_heatmap=outfname_lobes_heatmap,
            label=label,
            ignore_confidence=ignore_confidence,
            importance_type=importance_type,
            gain_axis_scale=gain_axis_scale,
            font_scale=font_scale,
        )

    else:
        show_plots = ss.args.get('show_plots', True)
        _plot_grouped_stacked_importance(
            outfname=outfname,
            feature_importance_3d=feature_importance_3d,
            group_labels=group_labels,
            freq_labels=freq_labels,
            parm_labels=parm_labels,
            target_label=label,
            ignore_confidence=ignore_confidence,
            importance_type=importance_type,
            gain_axis_scale=gain_axis_scale,
            font_scale=font_scale,
            figure_size_cfg=figure_size_cfg,
            y_limits_cfg=y_limits_cfg,
            show_plots=show_plots,
        )

        _plot_summary_by_group(
            outfname=outfname_group,
            feature_importance_3d=feature_importance_3d,
            group_labels=group_labels,
            target_label=label,
            ignore_confidence=ignore_confidence,
            importance_type=importance_type,
            gain_axis_scale=gain_axis_scale,
            font_scale=font_scale,
            figure_size_cfg=group_summary_figure_size_cfg,
            y_limits_cfg=group_summary_y_limits_cfg,
            show_plots=show_plots,
        )

        _plot_summary_by_frequency(
            outfname=outfname_freq,
            feature_importance_3d=feature_importance_3d,
            freq_labels=freq_labels,
            target_label=label,
            ignore_confidence=ignore_confidence,
            importance_type=importance_type,
            gain_axis_scale=gain_axis_scale,
            font_scale=font_scale,
            figure_size_cfg=freq_summary_figure_size_cfg,
            y_limits_cfg=freq_summary_y_limits_cfg,
            show_plots=show_plots,
        )

        _plot_summary_by_parameter(
            outfname=outfname_parm,
            feature_importance_3d=feature_importance_3d,
            parm_labels=parm_labels,
            target_label=label,
            ignore_confidence=ignore_confidence,
            importance_type=importance_type,
            gain_axis_scale=gain_axis_scale,
            font_scale=font_scale,
            figure_size_cfg=parm_summary_figure_size_cfg,
            y_limits_cfg=parm_summary_y_limits_cfg,
            show_plots=show_plots,
        )

    _append_feature_importance_summary_csv(
        ss=ss,
        fi_cfg=to_lobes_cfg,
        target_label=label,
        importance_type=importance_type,
        group_labels=ss.feature_importance_meta.get('group_labels', group_labels),
        freq_labels=freq_labels,
        parm_labels=parm_labels,
        feature_importance_3d=ss.feature_importance_3d,
    )


def _run_to_lobes_heatmap_branch(
    ss,
    feature_importance_3d,
    feature_names_3d,
    group_labels,
    freq_labels,
    parm_labels,
    lobes_heatmap_cfg,
    outfname_lobes_heatmap,
    label,
    ignore_confidence,
    importance_type,
    gain_axis_scale,
    font_scale,
):
    """Execute the use_lobes_heatmap=True branch for to_lobes feature-importance plots."""
    lobe_order_csv = ss.args['hosts'][ss.host].get('destrieux_lobes_order', None)

    if lobe_order_csv is None:
        raise KeyError(
            'Missing hosts.<host>.destrieux_lobes_order for to_lobes heatmap plotting'
        )

    ordered_lobes = _load_ordered_roi_names(lobe_order_csv)

    if len(ordered_lobes) == 0:
        raise ValueError('Empty lobe order loaded from destrieux_lobes_order CSV')

    group_to_idx = {name: idx for idx, name in enumerate(group_labels)}
    display_group_order = [name for name in ordered_lobes if name in group_to_idx]
    missing_in_order = [name for name in group_labels if name not in set(display_group_order)]

    if missing_in_order:
        print(
            'Warning: some lobe groups are missing in destrieux_lobes_order and will be '
            f'appended at the end: {missing_in_order}'
        )
        display_group_order.extend(missing_in_order)

    if len(display_group_order) == 0:
        raise ValueError(
            'No overlap between reducer group labels and destrieux_lobes_order CSV labels'
        )

    cmap_name = str(lobes_heatmap_cfg.get('color_scale', lobes_heatmap_cfg.get('colormap', 'mako')))
    limits_cfg = lobes_heatmap_cfg.get(
        'heatmap_limits',
        lobes_heatmap_cfg.get('limits', lobes_heatmap_cfg.get('color_limits', None)),
    )
    heatmap_gain_axis_scale = str(lobes_heatmap_cfg.get('gain_axis_scale', gain_axis_scale)).lower()
    median_y_limits_cfg = lobes_heatmap_cfg.get('median_y_limits', None)
    heatmap_y_axis_font_size_cfg = lobes_heatmap_cfg.get('heatmap_y_axis_font_size', None)
    figure_size_hm_cfg = lobes_heatmap_cfg.get('figure_size', None)
    panel_height_ratios_cfg = lobes_heatmap_cfg.get('panel_height_ratios', [3.0, 1.2])
    dpi_hm = int(lobes_heatmap_cfg.get('dpi', 300))

    if heatmap_gain_axis_scale not in ('linear', 'log'):
        raise ValueError(
            'Unsupported feature_importance.lobes_heatmap.gain_axis_scale. '
            'Allowed values are: ["linear", "log"]'
        )

    if heatmap_y_axis_font_size_cfg is None:
        heatmap_y_axis_font_size = None
    else:
        heatmap_y_axis_font_size = float(heatmap_y_axis_font_size_cfg)

        if heatmap_y_axis_font_size <= 0:
            raise ValueError('feature_importance.lobes_heatmap.heatmap_y_axis_font_size must be > 0')

    if dpi_hm <= 0:
        raise ValueError('feature_importance.lobes_heatmap.dpi must be > 0')

    if not isinstance(panel_height_ratios_cfg, (list, tuple)) or len(panel_height_ratios_cfg) != 2:
        raise ValueError(
            'feature_importance.lobes_heatmap.panel_height_ratios must be '
            '[heatmap_ratio, median_ratio]'
        )

    heat_ratio = float(panel_height_ratios_cfg[0])
    median_ratio = float(panel_height_ratios_cfg[1])

    if heat_ratio <= 0 or median_ratio <= 0:
        raise ValueError(
            'feature_importance.lobes_heatmap.panel_height_ratios values must be > 0'
        )

    display_row_idx = [group_to_idx[name] for name in display_group_order]
    feature_importance_3d_disp = feature_importance_3d[display_row_idx, :, :]
    feature_names_3d_disp = feature_names_3d[display_row_idx, :, :]
    nonzero_row_mask = np.any(feature_importance_3d_disp > 0.0, axis=(1, 2))

    plot_3d_heatmap(
        feature_importance_3d_disp,
        dim1_labels=display_group_order,
        dim2_labels=freq_labels,
        dim3_labels=parm_labels,
        outfname=outfname_lobes_heatmap,
        suptitle=(
            f'Lobe feature importance ({importance_type}) for target label: {label} '
            f'(ignore_confidence={ignore_confidence})'
        ),
        cfg={
            'show_plots': bool(ss.args.get('show_plots', True)),
            'style': str(lobes_heatmap_cfg.get('style', 'whitegrid')),
            'font_scale': float(lobes_heatmap_cfg.get('font_scale', font_scale)),
            'figure_size': figure_size_hm_cfg,
            'panel_height_ratios': [heat_ratio, median_ratio],
            'hspace': float(lobes_heatmap_cfg.get('hspace', 0.15)),
            'wspace': float(lobes_heatmap_cfg.get('wspace', 0.35)),
            'dpi': dpi_hm,
            'colormap': cmap_name,
            'color_limits': limits_cfg,
            'axis_scale': heatmap_gain_axis_scale,
            'median_row_mask': nonzero_row_mask,
            'median_y_limits': median_y_limits_cfg,
            'heatmap_y_axis_font_size': heatmap_y_axis_font_size,
            'x_axis_label': 'Frequency (Hz)',
            'y_axis_label': 'Lobe',
            'median_y_label': 'Median across lobes',
            'colorbar_label': f'{importance_type} ({heatmap_gain_axis_scale})',
            'x_tick_mode': 'round_frequency',
            'median_exclude_zeros': bool(lobes_heatmap_cfg.get('median_exclude_zeros', True)),
        },
    )

    ss.feature_importance_3d = feature_importance_3d_disp
    ss.feature_names_3d = feature_names_3d_disp
    ss.feature_importance_meta = {
        'importance_type': importance_type,
        'group_labels': display_group_order,
        'freq_labels': freq_labels,
        'parm_labels': parm_labels,
        'nonzero_importance_group_row_mask': nonzero_row_mask.tolist(),
        'mode': 'to_lobes_heatmap',
    }
    print(f'Lobe heatmap feature-importance plot saved to: {outfname_lobes_heatmap}')


def _run_feature_importance_ccv(
    ss,
    payload,
    reducer,
    feature_names,
    model,
    ccv_cfg,
    input_stem_suffix='',
):
    """
    Feature-importance plotting for CCV-like dim reductions. In fact, this
    function is NOT specific to the CCV case as such; rather it is assumed that
    features from the reduced set can still be represented as (ROI, frequency, parm)
    triples. It also relies on the feature names to be in this format:

    '<L or R>-<lobe>|ch<N>|X.ddHz|<stat-moment>', for example: 'R-T|ch146|55.00Hz|std'

    It expects the mapping from chN to the actual channel names to be in the 2nd column
    of CSV file in the source folder, whose name is given by ss['cwt_channels_order']; the
    the first column just contains chN's. Also, the order in which the channels
    are shown in the heatmap is given by the 1st column of CSV pointed to by
    [host]['destrieux_order'] key.

    Args:
        ss (obj): Application object containing runtime settings and paths.
        payload (dict): Classification payload loaded from pickle.
        reducer (dict): Dimensionality-reduction metadata.
        feature_names (Sequence[str]): Flattened model feature names.
        model (xgboost.XGBModel): Trained model reconstructed from payload.
        ccv_cfg (dict): Method-specific config under ``feature_importance.ccv``.
        input_stem_suffix (str): Optional stem suffix propagated from the input
            classifier pickle name (for example ``_ijob7``), appended to output
            plot file stems.

    Returns:
        None

    """
    importance_type = str(ccv_cfg.get('ranking', ss.args.get('feature_ranking', 'gain'))).lower()

    if importance_type not in _ALLOWED_IMPORTANCE_TYPES:
        raise ValueError(
            f'Unsupported feature_importance.ccv.ranking="{importance_type}". '
            f'Allowed values are: {sorted(_ALLOWED_IMPORTANCE_TYPES)}'
        )

    gain_axis_scale = str(ccv_cfg.get('gain_axis_scale', ss.args.get('gain_axis_scale', 'log'))).lower()

    if gain_axis_scale not in ('linear', 'log'):
        raise ValueError(
            'Unsupported feature_importance.ccv.gain_axis_scale. '
            'Allowed values are: ["linear", "log"]'
        )

    font_scale = float(ccv_cfg.get('font_scale', ss.args.get('feature_importance_font_scale', 0.75)))

    if font_scale <= 0:
        raise ValueError('feature_importance.ccv.font_scale must be > 0')

    cmap_name = str(ccv_cfg.get('color_scale', ccv_cfg.get('colormap', 'mako')))
    limits_cfg = ccv_cfg.get('heatmap_limits', ccv_cfg.get('limits', ccv_cfg.get('color_limits', None)))
    median_y_limits_cfg = ccv_cfg.get('median_y_limits', None)
    heatmap_y_axis_font_size_cfg = ccv_cfg.get('heatmap_y_axis_font_size', None)
    figure_size_cfg = ccv_cfg.get('figure_size', None)
    panel_height_ratios_cfg = ccv_cfg.get('panel_height_ratios', [3.0, 1.2])
    dpi = int(ccv_cfg.get('dpi', 150))

    if heatmap_y_axis_font_size_cfg is None:
        heatmap_y_axis_font_size = None
    else:
        heatmap_y_axis_font_size = float(heatmap_y_axis_font_size_cfg)

        if heatmap_y_axis_font_size <= 0:
            raise ValueError('feature_importance.ccv.heatmap_y_axis_font_size must be > 0')

    if dpi <= 0:
        raise ValueError('feature_importance.ccv.dpi must be > 0')

    if not isinstance(panel_height_ratios_cfg, (list, tuple)) or len(panel_height_ratios_cfg) != 2:
        raise ValueError(
            'feature_importance.ccv.panel_height_ratios must be [heatmap_ratio, median_ratio]'
        )

    heat_ratio = float(panel_height_ratios_cfg[0])
    median_ratio = float(panel_height_ratios_cfg[1])

    if heat_ratio <= 0 or median_ratio <= 0:
        raise ValueError('feature_importance.ccv.panel_height_ratios values must be > 0')

    if reducer.get('method', '') != 'consensus_cv':
        raise ValueError('CCV feature importance requires reducer method "consensus_cv"')

    # This is a vector of importance values for "feature_names"
    # Note that while those are most important features, the importances are NOT sorted.
    # This is because during consensus procedures the important feature indicies themselves
    # were sorted.
    feature_importance_1d = _extract_importance_vector(model, feature_names, importance_type)

    if feature_importance_1d.shape[0] != len(feature_names):
        raise RuntimeError('Feature importance length mismatch with feature_names')

    parsed = []

    # Note that for CCV we used the following feature name encoding:
    # "<hemi>-<lobe>|chN|<f>Hz|<parm_name>"
    for idx, name in enumerate(feature_names):
        parts = str(name).split('|')

        if len(parts) != 4:
            raise ValueError(
                f'Unexpected CCV feature name format: "{name}". '
                'Expected "lobe|ch<idx>|<freq>Hz|<parm>".'
            )

        ch_token = parts[1].strip()
        freq_token = parts[2].strip()
        parm_name = parts[3].strip()
        ch_match = re.fullmatch(r'ch(\d+)', ch_token)
        freq_match = re.fullmatch(r'(\d+\.\d{2})Hz', freq_token)

        if ch_match is None:
            raise ValueError(f'Cannot parse channel token "{ch_token}" in feature "{name}"')

        if freq_match is None:
            raise ValueError(f'Cannot parse frequency token "{freq_token}" in feature "{name}"')

        ch_idx = int(ch_match.group(1))
        freq_hz = float(freq_match.group(1))
        parsed.append((idx, ch_idx, freq_hz, parm_name, str(name)))

    # Now "parsed" is a list of tuples (feature_no, chN, freq, parm-name, <full-feature-name>)

    # Preserve first-seen order for parameter labels to match training feature naming. Due
    # to CCV reducer returning features in order of significance, the parameter labels for
    # plotting will also be ordered by significance.
    parm_labels = []
    parm_seen = set()

    for _, _, _, parm_name, _ in parsed:
        if parm_name not in parm_seen:
            parm_seen.add(parm_name)
            parm_labels.append(parm_name)

    if len(parm_labels) == 0:
        raise ValueError('No parameter labels found in CCV feature names')

    # Among the CCV features used by the reducer some parameters may be missing (as not important).
    # Display warning.
    expected_nparms = 5 if bool(payload.get('use_moments_only', ss.args.get('use_moments_only', False))) else 9

    if len(parm_labels) != expected_nparms:
        print(
            f'\nWarning: only nparms={len(parm_labels)} parameters are in use by selected CCV features,\n'
            f'while the full feature set uses {expected_nparms} parameters.\n'
        )

    order_csv = ss.args['hosts'][ss.host].get('destrieux_order', None)

    if order_csv is None:
        raise KeyError('Missing hosts.<host>.destrieux_order for CCV feature-importance plotting')

    # roi_order lists ROIs so that geographic vicinity is preserved as much as possible
    # Note that channels are ordered *differently* in CWT files - see ch_idx_to_roi_name
    # below.
    roi_order = _load_ordered_roi_names(order_csv)  # Just an ordered list of atlas ROIs

    if len(roi_order) == 0:
        raise ValueError('Empty ROI order loaded from destrieux_order CSV')

    # Load cwt_channels_order CSV: maps ch<N> index -> actual ROI name for the
    # channels that were used during training.  nchans_used may be smaller than
    # len(roi_order) when not all atlas ROIs are present in the source data.
    cwt_ch_order_key = ss.args.get('cwt_channels_order', None)

    if cwt_ch_order_key is None:
        raise KeyError('Missing top-level "cwt_channels_order" key in configuration')

    data_root = ss.args['hosts'][ss.host]['sources']['data_root']
    cwt_ch_order_csv = Path(data_root) / cwt_ch_order_key

    # ch_idx_to_roi_name is mapping ch # -> ROI as used in CWT files;
    # this ordering is NOT that of roi_order
    ch_idx_to_roi_name = _load_cwt_channels_order(cwt_ch_order_csv)
    nchans_used = len(ch_idx_to_roi_name)

    if nchans_used == 0:
        raise ValueError(f'Empty cwt_channels_order CSV: {cwt_ch_order_csv}')

    # Build the display row order: ROIs that were actually used, sorted by their
    # position in the destrieux_order atlas CSV so the heatmap matches the atlas.
    used_roi_set = set(ch_idx_to_roi_name.values())
    display_roi_order = [name for name in roi_order if name in used_roi_set]

    # display_roi_order is just a list of ROIs actually used in CWTs ordered in
    # accordance with roi_order
    if len(display_roi_order) == 0:
        raise ValueError(
            'No overlap between cwt_channels_order ROI names and destrieux_order ROI names'
        )

    # Create mapping ROI -> heat map row number
    roi_name_to_display_row = {name: i for i, name in enumerate(display_roi_order)}

    freq_values, freq_to_idx = _resolve_ccv_frequency_axis(
        reducer=reducer,
        parsed=parsed,
        ccv_cfg=ccv_cfg,
        nchans=nchans_used,
        expected_nparms=expected_nparms,
    )

    nfreqs = len(freq_values)
    freq_labels = [f'{frq:g}Hz' for frq in freq_values]

    nrows = len(display_roi_order)

    # This is the will be 3d feature array with proper ROI ordering
    # Note that it can't be obtained by wrapping back 1d features, because
    # the order is different and some of ROIs, freqs and parms may be missing.
    feature_importance_3d = np.zeros((nrows, nfreqs, len(parm_labels)), dtype=float)
    feature_names_3d = np.full((nrows, nfreqs, len(parm_labels)), '', dtype=object)

    parm_to_idx = {name: ip for ip, name in enumerate(parm_labels)}

    unknown_ch = set()  # Used only to display warning if some chN do not have
                        # corresponding ROI

    encountered_ch_tokens = sorted({f'ch{ch_idx}' for _, ch_idx, _, _, _ in parsed}, key=lambda x: int(x[2:]))

    # If a set of most important features was chosen too generously,
    # some of them may have zero importance. We do not want those
    # to be counted im median scores calculations.
    nonzero_ch_tokens = set()   # This is a set of 'chN's that correspond
                                # features with non-zero importances
    nonzero_row_mask = np.zeros(nrows, dtype=bool)

    # This convoluted cycle places rows in the heatmap in desired order
    # as in roi_order list
    for feat_idx, ch_idx, frq, parm_name, fname in parsed:
        iparm = parm_to_idx[parm_name]
        ifreq = freq_to_idx[frq]
        roi_name = ch_idx_to_roi_name.get(ch_idx, None)

        if roi_name is None:
            unknown_ch.add(ch_idx)
            continue

        # Here we are using the mapping ROI name -> heat map row number
        irow = roi_name_to_display_row.get(roi_name, None)

        if irow is None:
            # ROI exists in cwt_channels_order but is absent from destrieux_order.
            continue

        importance_val = float(feature_importance_1d[feat_idx])

        if importance_val > 0.0:
            nonzero_row_mask[irow] = True
            nonzero_ch_tokens.add(f'ch{ch_idx}')

        # Plug the importance for this feature into correct cell of
        # feature_importance_3d
        feature_importance_3d[irow, ifreq, iparm] = importance_val
        feature_names_3d[irow, ifreq, iparm] = fname

    # The status now:
    #   - feature_importance_3d, feature_names_3d are ready go use for plotting
    #   - nonzero_row_mask is mask for the rows in heatmap that correspond
    #     to non-zero feature importances
    print(f'{np.sum(nonzero_row_mask)} out of {len(nonzero_row_mask)} ROIs have non-zero importances.')

    if unknown_ch:
        print(
            f'Warning: {len(unknown_ch)} ch indices from feature names '
            f'not found in cwt_channels_order CSV: {sorted(unknown_ch)}'
        )

    label = str(payload.get('target_label', ss.args.get('target_label', 'label')))
    ignore_confidence = bool(ss.args.get('ignore_confidence', False))
    outfname = Path(ss.out_root) / (
        f'feature_importance_ccv_{label}_{importance_type}_ignoreConf{ignore_confidence}'
        f'{input_stem_suffix}.png'
    )

    plot_3d_heatmap(
        feature_importance_3d,
        dim1_labels=display_roi_order,
        dim2_labels=freq_labels,
        dim3_labels=parm_labels,
        outfname=outfname,
        suptitle=(
            f'CCV feature importance ({importance_type}) for target label: {label} '
            f'(ignore_confidence={ignore_confidence})'
        ),
        cfg={
            'show_plots': bool(ss.args.get('show_plots', True)),
            'style': 'whitegrid',
            'font_scale': font_scale,
            'figure_size': figure_size_cfg,
            'panel_height_ratios': [heat_ratio, median_ratio],
            'hspace': 0.15,
            'wspace': 0.35,
            'dpi': dpi,
            'colormap': cmap_name,
            'color_limits': limits_cfg,
            'axis_scale': gain_axis_scale,
            'median_row_mask': nonzero_row_mask,
            'median_y_limits': median_y_limits_cfg,
            'heatmap_y_axis_font_size': heatmap_y_axis_font_size,
            'x_axis_label': 'Frequency (Hz)',
            'y_axis_label': 'ROI',
            'median_y_label': 'Median across ROI',
            'colorbar_label': f'{importance_type} ({gain_axis_scale})',
            'x_tick_mode': 'round_frequency',
        },
    )

    ss.feature_importance_1d = feature_importance_1d
    ss.feature_importance_3d = feature_importance_3d
    ss.feature_names_3d = feature_names_3d
    ss.feature_importance_meta = {
        'importance_type': importance_type,
        'group_labels': display_roi_order,
        'freq_labels': freq_labels,
        'parm_labels': parm_labels,
        'encountered_channels': encountered_ch_tokens,
        'nonzero_importance_channels': sorted(nonzero_ch_tokens, key=lambda x: int(x[2:])),
        'nonzero_importance_roi_row_mask': nonzero_row_mask.tolist(),
        'nonzero_importance_roi_labels': [
            display_roi_order[idx]
            for idx, is_used in enumerate(nonzero_row_mask)
            if is_used
        ],
        'mode': 'consensus_cv',
    }

    print(f'CCV feature-importance plot saved to: {outfname}')


def _resolve_ccv_frequency_axis(reducer, parsed, ccv_cfg, nchans, expected_nparms):
    """
    Resolve full frequency axis for CCV plots and map reduced-feature frequencies
    into that full axis.

    Args:
        reducer (dict): CCV reducer metadata.
        parsed (list[tuple]): Parsed feature tuples from CCV names.
        ccv_cfg (dict): ``feature_importance.ccv`` configuration dictionary.
        nchans (int): Number of ROI channels in display order.
        expected_nparms (int): Number of parameters in the full feature space.

    Returns:
        tuple[list[float], dict[float, int]]: Full frequency values and mapping
        from reduced-feature frequency values to full-axis indices.

    """
    # Infer expected full frequency count from the original (pre-CCV) feature space.
    n_features_in = reducer.get('n_features_in', None)

    if n_features_in is None:
        raise KeyError(
            'CCV reducer metadata is missing "n_features_in"; cannot infer full number of frequencies'
        )

    denom = int(nchans) * int(expected_nparms)

    if denom <= 0 or int(n_features_in) % denom != 0:
        raise ValueError(
            'Cannot infer full number of frequencies from reducer metadata: '
            f'n_features_in={n_features_in}, nchans={nchans}, nparms={expected_nparms}'
        )

    nfreqs_full = int(n_features_in) // denom

    if nfreqs_full <= 0:
        raise ValueError(f'Inferred invalid full frequency count: {nfreqs_full}')

    freq_values_reduced = sorted({item[2] for item in parsed})
    nfreqs_reduced = len(freq_values_reduced)

    if nfreqs_reduced == 0:
        raise ValueError('No frequencies found in CCV feature names')

    if nfreqs_reduced == nfreqs_full:
        # All full-set frequencies are represented by selected CCV features.
        freq_values = freq_values_reduced
        freq_to_idx = {frq: ifreq for ifreq, frq in enumerate(freq_values)}
        return freq_values, freq_to_idx

    # Reduced set misses some frequencies: require explicit full frequency list.
    all_freqs_cfg = ccv_cfg.get('all_freqs', None)

    if all_freqs_cfg is None:
        raise ValueError(
            'Reduced CCV feature set does not contain all full-set frequencies '
            f'(reduced={nfreqs_reduced}, expected={nfreqs_full}). '
            'Please provide feature_importance.ccv.all_freqs.'
        )

    if not isinstance(all_freqs_cfg, (list, tuple)):
        raise ValueError('feature_importance.ccv.all_freqs must be a list of frequencies')

    if len(all_freqs_cfg) != nfreqs_full:
        raise ValueError(
            'feature_importance.ccv.all_freqs length mismatch: '
            f'len(all_freqs)={len(all_freqs_cfg)} but expected {nfreqs_full}'
        )

    freq_values = [float(frq) for frq in all_freqs_cfg]

    # Match reduced frequencies encoded in feature names (rounded to 2 decimals)
    # to the full-frequency list supplied in all_freqs.
    full_key_to_idx = {}

    for ifreq, frq in enumerate(freq_values):
        key = f'{float(frq):.2f}'

        if key in full_key_to_idx:
            raise ValueError(
                'feature_importance.ccv.all_freqs is ambiguous after rounding to 2 decimals '
                f'(duplicate key {key})'
            )

        full_key_to_idx[key] = ifreq

    freq_to_idx = {}

    for frq in freq_values_reduced:
        key = f'{float(frq):.2f}'

        if key not in full_key_to_idx:
            raise ValueError(
                f'Reduced feature frequency {frq:g}Hz is not represented in '
                'feature_importance.ccv.all_freqs'
            )

        freq_to_idx[frq] = full_key_to_idx[key]

    return freq_values, freq_to_idx


def _load_ordered_roi_names(order_csv_path):
    """
    Load ROI names from the ``name`` column of an atlas-order CSV file.

    Args:
        order_csv_path (str | pathlib.Path): Path to CSV with a ``name`` column.

    Returns:
        list[str]: ROI names in the exact order provided by CSV rows.

    """
    p = Path(order_csv_path)

    if not p.exists():
        raise FileNotFoundError(f'destrieux_order CSV not found: {p}')

    names = []

    with open(p, 'r', newline='') as fp:
        reader = csv.DictReader(fp)

        if reader.fieldnames is None or 'name' not in reader.fieldnames:
            raise ValueError(f'destrieux_order CSV must contain a "name" column: {p}')

        for row in reader:
            nm = str(row.get('name', '')).strip()

            if nm:
                names.append(nm)

    if not names:
        raise ValueError(f'No ROI names found in destrieux_order CSV: {p}')

    return names


def _load_cwt_channels_order(csv_path):
    """
    Load the ``ch<N>`` → ROI-name mapping produced by the CWT channel-order CSV.

    Args:
        csv_path (str | pathlib.Path): Path to CSV with ``ch_num`` and ``name`` columns.

    Returns:
        dict[int, str]: Mapping from 0-based channel index to ROI name.

    """
    p = Path(csv_path)

    if not p.exists():
        raise FileNotFoundError(f'cwt_channels_order CSV not found: {p}')

    mapping = {}

    with open(p, 'r', newline='') as fp:
        reader = csv.DictReader(fp)

        if reader.fieldnames is None or 'ch_num' not in reader.fieldnames or 'name' not in reader.fieldnames:
            raise ValueError(
                f'cwt_channels_order CSV must contain "ch_num" and "name" columns: {p}'
            )

        for row in reader:
            ch_token = str(row.get('ch_num', '')).strip()
            roi_name = str(row.get('name', '')).strip()

            if not ch_token or not roi_name:
                continue

            m = re.fullmatch(r'ch(\d+)', ch_token)

            if m is None:
                raise ValueError(
                    f'Unexpected ch_num value "{ch_token}" in cwt_channels_order CSV: {p}'
                )

            mapping[int(m.group(1))] = roi_name

    if not mapping:
        raise ValueError(f'No entries loaded from cwt_channels_order CSV: {p}')

    return mapping


def _top_labels_by_values(labels, values, n_top):
    """
    Return the labels associated with the largest importance values.

    Args:
        labels (Sequence[str]): Labels aligned with ``values``.
        values (array-like): Numeric values used to rank ``labels``.
        n_top (int): Maximum number of top-ranked labels to return.

    Returns:
        list[str]: Labels sorted by descending value, truncated to ``n_top``.

    """
    if n_top <= 0:
        return []

    labels = list(labels)
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return []

    n = min(int(n_top), len(labels), int(values.size))
    ranked_idx = np.argsort(values)[::-1]
    return [str(labels[idx]) for idx in ranked_idx[:n]]


def _append_feature_importance_summary_csv(
    ss,
    fi_cfg,
    target_label,
    importance_type,
    group_labels,
    freq_labels,
    parm_labels,
    feature_importance_3d,
):
    """
    Append a one-row feature-importance summary to the configured CSV file.

    Args:
        ss (obj): Application object providing output paths and classifier args.
        fi_cfg (dict): Feature-importance configuration dictionary.
        target_label (str): Classifier target label associated with the summary.
        importance_type (str): XGBoost importance metric used for ranking.
        group_labels (Sequence[str]): Lobe or group labels for axis 0.
        freq_labels (Sequence[str]): Frequency labels for axis 1.
        parm_labels (Sequence[str]): Parameter labels for axis 2.
        feature_importance_3d (numpy.ndarray): Importance array with shape
            ``(ngroups, nfreqs, nparms)``.

    Returns:
        None

    """
    summary_csv_name = fi_cfg.get('summary_csv_name', 'feature_importance_summary.csv')
    n_top_lobes = int(fi_cfg.get('n_top_lobes', 5))
    n_top_freqs = int(fi_cfg.get('n_top_freqs', 5))
    n_top_parms = int(fi_cfg.get('n_top_parms', 5))

    if n_top_lobes < 0 or n_top_freqs < 0 or n_top_parms < 0:
        raise ValueError('feature_importance n_top_lobes/n_top_freqs/n_top_parms must be >= 0')

    lobe_vals = feature_importance_3d.sum(axis=(1, 2)).astype(float)
    freq_vals = feature_importance_3d.sum(axis=(0, 2)).astype(float)
    parm_vals = feature_importance_3d.sum(axis=(0, 1)).astype(float)

    formatted_freq_labels = _format_frequency_labels(freq_labels)
    top_lobes = _top_labels_by_values(group_labels, lobe_vals, n_top_lobes)
    top_freqs = _top_labels_by_values(formatted_freq_labels, freq_vals, n_top_freqs)
    top_parms = _top_labels_by_values(parm_labels, parm_vals, n_top_parms)

    hospitals = ss.args.get('hospital', [])

    if isinstance(hospitals, list):
        hospital_val = ss.hlist(hospitals)
    else:
        hospital_val = str(hospitals)

    csv_path = Path(summary_csv_name)

    if not csv_path.is_absolute():
        csv_path = Path(ss.out_root) / csv_path

    row = {
        'hospital': hospital_val,
        'target_label': str(target_label),
        'use_moments_only': bool(ss.args.get('use_moments_only', False)),
        'ignore_confidence': bool(ss.args.get('ignore_confidence', False)),
        'standardize_features': bool(ss.args.get('standardize_features', False)),
        'ranking': str(importance_type),
        'lobes': '|'.join(top_lobes),
        'freqs': '|'.join(top_freqs),
        'parms': '|'.join(top_parms),
    }

    fieldnames = [
        'hospital',
        'target_label',
        'use_moments_only',
        'ignore_confidence',
        'standardize_features',
        'ranking',
        'lobes',
        'freqs',
        'parms',
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    print(f'Feature-importance summary appended to: {csv_path}')


def _find_results_pickle(ss):
    """
    Locate the classification-results pickle corresponding to current JSON settings.

    Args:
        ss (obj): Application object containing classifier settings and output
            path helpers.

    Returns:
        pathlib.Path: Path to the expected or best-matching classifier pickle.

    """
    label = ss.args['target_label']
    standardize = ss.args.get('standardize_features', False)
    ignore_confidence = ss.args.get('ignore_confidence', False)
    nparms = 5 if ss.args.get('use_moments_only', False) else 9
    hlist = ss.hlist(ss.args['hospital'])

    expected = ss.cls_pkl_pname(hlist, label, nparms, standardize, ignore_confidence)

    if expected.exists():
        return expected

    if bool(ss.args.get('use_consensus_cv', False)):
        dim_red = '_ccv'
    else:
        dim_method = str(ss.args.get('dim_reduction', '')).lower()

        if dim_method == 'epi_features':
            dim_red = '_epi'
        else:
            # to_lobes (and default/legacy settings) use no reducer suffix.
            dim_red = ''

    pattern = f'xgb_*_{label}{dim_red}_nparms*_std*_ignoreConf{ignore_confidence}*.pkl'
    candidates = sorted(ss.out_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(
            f'No classification pickle found for label="{label}" in {ss.out_root} '
            f'using expected file {expected.name} or pattern {pattern}'
        )

    print(f'Warning: expected file not found, using latest match: {candidates[0].name}')
    return candidates[0]


def _extract_ijob_suffix(pkl_pname):
    """Return trailing batch-job suffix from pickle stem, e.g. ``_ijob7``."""
    match = re.search(r'(_ijob\d+)$', Path(pkl_pname).stem)
    return match.group(1) if match else ''


def _extract_importance_vector(model, feature_names, importance_type):
    """
    Return importance values aligned to `feature_names` list.

    Args:
        model (xgboost.XGBModel): Trained XGBoost model wrapper.
        feature_names (Sequence[str]): Feature names in the desired output order.
        importance_type (str): XGBoost importance metric name passed to
            ``get_score``.

    Returns:
        numpy.ndarray: One-dimensional importance vector aligned with
        ``feature_names``.

    """
    # Get raw importance values from XGBoost booster.
    booster = model.get_booster()

    # scores is a dict returned by xgboost which maps feature_name -> score value)
    scores = booster.get_score(importance_type=importance_type)

    # Preferred path: score entries are keyed by the actual feature names.
    # Build output in exact feature_names order and fill missing entries with 0.
    values = np.array([float(scores.get(name, 0.0)) for name in feature_names], dtype=float)

    # For tree-based boosters only non-negative scores are expected.
    # If at least one named match exists, the feature names -> score mapping is most
    # likely ok and fallback path is not needed. This will usually be true.
    if np.any(values > 0):
        return values

    # Fallback path: some models store scores as positional keys like f0, f1, ...
    # Start from zeros and map each positional key into the corresponding index.
    values = np.zeros(len(feature_names), dtype=float)

    for key, val in scores.items():
        # Accept only positional key format and ignore anything else.
        match = re.fullmatch(r'f(\d+)', str(key))

        if match is None:
            continue

        idx = int(match.group(1))

        # Guard against malformed indices that fall outside feature_names range.
        if 0 <= idx < len(values):
            values[idx] = float(val)

    # Return aligned vector regardless of whether non-zero scores were found.
    return values


def _infer_to_lobes_dims(feature_names, group_labels):
    """
    Infer dimensions and labels for flattened to_lobes feature names.

    Args:
        feature_names (Sequence[str]): Flattened feature names expected in
            ``group|freq|parm`` format.
        group_labels (Sequence[str]): Ordered lobe or group labels from the
            reducer metadata.

    Returns:
        tuple[int, int, int, list[str], list[str]]: Number of groups,
        frequencies, parameters, plus inferred frequency and parameter labels.

    """
    ngroups = len(group_labels)
    nfeatures = len(feature_names)

    if ngroups == 0:
        raise ValueError('Reducer has no group labels; cannot restore to_lobes feature layout')

    if nfeatures % ngroups != 0:
        raise ValueError(
            f'Feature count {nfeatures} is not divisible by #groups {ngroups}; cannot reshape'
        )

    split_names = [name.split('|') for name in feature_names]

    if not all(len(parts) == 3 for parts in split_names):
        raise ValueError('Feature names are not in expected "group|freq|parm" format for to_lobes')

    parm_labels = []
    parm_seen = set()

    for _, _, parm in split_names:
        if parm not in parm_seen:
            parm_seen.add(parm)
            parm_labels.append(parm)

    nparms = len(parm_labels)

    if nparms == 0:
        raise ValueError('Could not infer parameter labels from feature names')

    features_per_group = nfeatures // ngroups

    if features_per_group % nparms != 0:
        raise ValueError(
            f'Features/group={features_per_group} is not divisible by nparms={nparms}; cannot infer nfreqs'
        )

    nfreqs = features_per_group // nparms

    first_group_names = split_names[:features_per_group]
    freq_labels = []
    freq_seen = set()

    for _, freq, _ in first_group_names:
        if freq not in freq_seen:
            freq_seen.add(freq)
            freq_labels.append(freq)

    if len(freq_labels) != nfreqs:
        raise ValueError(
            f'Inferred nfreqs={nfreqs}, but found {len(freq_labels)} unique frequencies in first group'
        )

    return ngroups, nfreqs, nparms, freq_labels, parm_labels


def _resolve_figure_size(figure_size_cfg, ncols, nrows):
    """
    Resolve figure size in inches.

    Args:
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            user-specified figure size.
        ncols (int): Number of subplot columns.
        nrows (int): Number of subplot rows.

    Returns:
        tuple[float, float]: Figure width and height in inches.

    Notes:
        ``figure_size_cfg`` can be ``None`` to derive a landscape-oriented size,
        or a two-element sequence ``[width, height]`` with positive values.

    """
    if figure_size_cfg is None:
        width = max(10.0, 4.0 * ncols)
        height = max(4.0, 3.4 * nrows)

        if width <= height:
            width = max(width, height * 1.2)

        return width, height

    if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
        raise ValueError('feature_importance.figure_size must be null or [width, height]')

    width = float(figure_size_cfg[0])
    height = float(figure_size_cfg[1])

    if width <= 0 or height <= 0:
        raise ValueError('feature_importance.figure_size values must be > 0')

    return width, height


def _resolve_summary_figure_size(figure_size_cfg, default_width=12.0, default_height=6.0):
    """
    Resolve figure size for single-panel summary plots.

    Args:
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            user-specified summary figure size.
        default_width (float): Default width to use when no explicit size is
            provided.
        default_height (float): Default height to use when no explicit size is
            provided.

    Returns:
        tuple[float, float]: Summary figure width and height in inches.

    """
    if figure_size_cfg is None:
        width = default_width
        height = default_height

        if width <= height:
            width = max(width, height * 1.2)

        return width, height

    if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
        raise ValueError('feature_importance summary figure size must be null or [width, height]')

    width = float(figure_size_cfg[0])
    height = float(figure_size_cfg[1])

    if width <= 0 or height <= 0:
        raise ValueError('Summary figure size values must be > 0')

    return width, height


def _plot_summary_by_group(
    outfname,
    feature_importance_3d,
    group_labels,
    target_label,
    ignore_confidence,
    importance_type,
    gain_axis_scale,
    font_scale,
    figure_size_cfg,
    y_limits_cfg,
    show_plots,
):
    """
    Plot total feature importance per lobe (group) - summed across frequencies and params.

    Args:
        outfname (pathlib.Path): Output path for the rendered figure.
        feature_importance_3d (numpy.ndarray): Importance array with shape
            ``(ngroups, nfreqs, nparms)``.
        group_labels (Sequence[str]): Labels for the group axis.
        target_label (str): Classifier target label shown in the title.
        ignore_confidence (bool): Whether confidence values were ignored in
            classifier inputs.
        importance_type (str): Importance metric name used for labeling.
        gain_axis_scale (str): Axis scaling mode for gain plots.
        font_scale (float): Seaborn font scaling factor.
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            figure size override.
        y_limits_cfg (list[float | None] | tuple[float | None, float | None] | None):
            Optional y-axis limits override.
        show_plots (bool): Whether to display the figure interactively.

    Returns:
        None

    """
    sns.set_theme(style='whitegrid', font_scale=font_scale)

    vals = feature_importance_3d.sum(axis=(1, 2)).astype(float)
    use_log_gain = (importance_type == 'gain') and (gain_axis_scale == 'log')
    eps = 1e-12
    plot_vals = np.clip(vals, eps, None) if use_log_gain else vals

    if use_log_gain:
        positive_vals = plot_vals[plot_vals > 0]

        if positive_vals.size > 0:
            auto_ymin = max(float(np.min(positive_vals)) * 0.8, eps)
            auto_ymax = max(float(np.max(positive_vals)) * 1.1, auto_ymin * 10.0)
        else:
            auto_ymin = eps
            auto_ymax = 1.0
    else:
        auto_ymin = 0.0
        auto_ymax = max(float(np.max(plot_vals)) * 1.05, 1e-9)

    ymin, ymax = _resolve_y_limits(y_limits_cfg, auto_ymin, auto_ymax, use_log_gain)
    fig_w, fig_h = _resolve_summary_figure_size(figure_size_cfg)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(group_labels))
    ax.bar(x, plot_vals, color=sns.color_palette('deep', 1)[0], width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=0, ha='center')
    ax.set_xlabel('Lobe')
    ax.set_ylabel(f'{importance_type} (log)' if use_log_gain else importance_type)
    ax.set_title(
        f'Lobe aggregated importance ({importance_type}) for label: {target_label} '
        f'(ignore_confidence={ignore_confidence})'
    )

    if use_log_gain:
        ax.set_yscale('log')

    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    outfname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfname, dpi=150, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f'Lobe aggregated importance plot saved to: {outfname}')


def _plot_summary_by_frequency(
    outfname,
    feature_importance_3d,
    freq_labels,
    target_label,
    ignore_confidence,
    importance_type,
    gain_axis_scale,
    font_scale,
    figure_size_cfg,
    y_limits_cfg,
    show_plots,
):
    """
    Plot total feature importance per frequency (summed across groups and params).

    Args:
        outfname (pathlib.Path): Output path for the rendered figure.
        feature_importance_3d (numpy.ndarray): Importance array with shape
            ``(ngroups, nfreqs, nparms)``.
        freq_labels (Sequence[str]): Labels for the frequency axis.
        target_label (str): Classifier target label shown in the title.
        ignore_confidence (bool): Whether confidence values were ignored in
            classifier inputs.
        importance_type (str): Importance metric name used for labeling.
        gain_axis_scale (str): Axis scaling mode for gain plots.
        font_scale (float): Seaborn font scaling factor.
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            figure size override.
        y_limits_cfg (list[float | None] | tuple[float | None, float | None] | None):
            Optional y-axis limits override.
        show_plots (bool): Whether to display the figure interactively.

    Returns:
        None

    """
    sns.set_theme(style='whitegrid', font_scale=font_scale)

    vals = feature_importance_3d.sum(axis=(0, 2)).astype(float)
    use_log_gain = (importance_type == 'gain') and (gain_axis_scale == 'log')
    eps = 1e-12
    plot_vals = np.clip(vals, eps, None) if use_log_gain else vals

    if use_log_gain:
        positive_vals = plot_vals[plot_vals > 0]

        if positive_vals.size > 0:
            auto_ymin = max(float(np.min(positive_vals)) * 0.8, eps)
            auto_ymax = max(float(np.max(positive_vals)) * 1.1, auto_ymin * 10.0)
        else:
            auto_ymin = eps
            auto_ymax = 1.0
    else:
        auto_ymin = 0.0
        auto_ymax = max(float(np.max(plot_vals)) * 1.05, 1e-9)

    ymin, ymax = _resolve_y_limits(y_limits_cfg, auto_ymin, auto_ymax, use_log_gain)
    fig_w, fig_h = _resolve_summary_figure_size(figure_size_cfg)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(freq_labels))
    ax.bar(x, plot_vals, color=sns.color_palette('deep', 1)[0], width=0.85)

    freq_tick_labels = _format_frequency_labels(freq_labels)
    round_tick_pos, round_tick_labels = _round_frequency_ticks(freq_labels)

    if round_tick_pos is not None:
        major_tick_pos = round_tick_pos
        major_tick_labels = round_tick_labels
    else:
        major_tick_pos = _major_tick_positions(len(freq_labels))
        major_tick_labels = [freq_tick_labels[idx] for idx in major_tick_pos]

    ax.set_xticks(major_tick_pos)
    ax.set_xticklabels(major_tick_labels, rotation=0, ha='center')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'{importance_type} (log)' if use_log_gain else importance_type)
    ax.set_title(
        f'Frequency aggregated importance ({importance_type}) for label: {target_label} '
        f'(ignore_confidence={ignore_confidence})'
    )

    if use_log_gain:
        ax.set_yscale('log')

    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    outfname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfname, dpi=150, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f'Frequency aggregated importance plot saved to: {outfname}')


def _plot_summary_by_parameter(
    outfname,
    feature_importance_3d,
    parm_labels,
    target_label,
    ignore_confidence,
    importance_type,
    gain_axis_scale,
    font_scale,
    figure_size_cfg,
    y_limits_cfg,
    show_plots,
):
    """
    Plot total feature importance per parameter (summed across groups and frequencies).

    Args:
        outfname (pathlib.Path): Output path for the rendered figure.
        feature_importance_3d (numpy.ndarray): Importance array with shape
            ``(ngroups, nfreqs, nparms)``.
        parm_labels (Sequence[str]): Labels for the parameter axis.
        target_label (str): Classifier target label shown in the title.
        ignore_confidence (bool): Whether confidence values were ignored in
            classifier inputs.
        importance_type (str): Importance metric name used for labeling.
        gain_axis_scale (str): Axis scaling mode for gain plots.
        font_scale (float): Seaborn font scaling factor.
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            figure size override.
        y_limits_cfg (list[float | None] | tuple[float | None, float | None] | None):
            Optional y-axis limits override.
        show_plots (bool): Whether to display the figure interactively.

    Returns:
        None

    """
    sns.set_theme(style='whitegrid', font_scale=font_scale)

    vals = feature_importance_3d.sum(axis=(0, 1)).astype(float)
    use_log_gain = (importance_type == 'gain') and (gain_axis_scale == 'log')
    eps = 1e-12
    plot_vals = np.clip(vals, eps, None) if use_log_gain else vals

    if use_log_gain:
        positive_vals = plot_vals[plot_vals > 0]

        if positive_vals.size > 0:
            auto_ymin = max(float(np.min(positive_vals)) * 0.8, eps)
            auto_ymax = max(float(np.max(positive_vals)) * 1.1, auto_ymin * 10.0)
        else:
            auto_ymin = eps
            auto_ymax = 1.0
    else:
        auto_ymin = 0.0
        auto_ymax = max(float(np.max(plot_vals)) * 1.05, 1e-9)

    ymin, ymax = _resolve_y_limits(y_limits_cfg, auto_ymin, auto_ymax, use_log_gain)
    fig_w, fig_h = _resolve_summary_figure_size(figure_size_cfg)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(parm_labels))
    ax.bar(x, plot_vals, color=sns.color_palette('deep', 1)[0], width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(parm_labels, rotation=0, ha='center')
    ax.set_xlabel('Parameter')
    ax.set_ylabel(f'{importance_type} (log)' if use_log_gain else importance_type)
    ax.set_title(
        f'Parameter aggregated importance ({importance_type}) for label: {target_label} '
        f'(ignore_confidence={ignore_confidence})'
    )

    if use_log_gain:
        ax.set_yscale('log')

    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    outfname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfname, dpi=150, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f'Parameter aggregated importance plot saved to: {outfname}')


def _plot_grouped_stacked_importance(
    outfname,
    feature_importance_3d,
    group_labels,
    freq_labels,
    parm_labels,
    target_label,
    ignore_confidence,
    importance_type,
    gain_axis_scale,
    font_scale,
    figure_size_cfg,
    y_limits_cfg,
    show_plots,
):
    """
    Plot one subplot per group (lobe): x-axis frequencies, stacked bars over parameters.

    Args:
        outfname (pathlib.Path): Output path for the rendered figure.
        feature_importance_3d (numpy.ndarray): Importance array with shape
            ``(ngroups, nfreqs, nparms)``.
        group_labels (Sequence[str]): Labels for the group axis.
        freq_labels (Sequence[str]): Labels for the frequency axis.
        parm_labels (Sequence[str]): Labels for the stacked parameter axis.
        target_label (str): Classifier target label shown in the title.
        ignore_confidence (bool): Whether confidence values were ignored in
            classifier inputs.
        importance_type (str): Importance metric name used for labeling.
        gain_axis_scale (str): Axis scaling mode for gain plots.
        font_scale (float): Seaborn font scaling factor.
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            figure size override.
        y_limits_cfg (list[float | None] | tuple[float | None, float | None] | None):
            Optional y-axis limits override.
        show_plots (bool): Whether to display the figure interactively.

    Returns:
        None

    """
    sns.set_theme(style='whitegrid', font_scale=font_scale)

    ngroups, nfreqs, nparms = feature_importance_3d.shape
    ncols = min(4, ngroups)
    nrows = int(np.ceil(ngroups / ncols))
    last_row_is_full = (ngroups % ncols == 0)

    if last_row_is_full or nrows == 1:
        rows_with_freq_ticks = {nrows - 1}
    else:
        rows_with_freq_ticks = {nrows - 2, nrows - 1}

    fig_w, fig_h = _resolve_figure_size(figure_size_cfg, ncols, nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    palette = sns.color_palette('deep', nparms)
    x = np.arange(nfreqs)
    freq_tick_labels = _format_frequency_labels(freq_labels)
    round_tick_pos, round_tick_labels = _round_frequency_ticks(freq_labels)

    if round_tick_pos is not None:
        major_tick_pos = round_tick_pos
        major_tick_labels = round_tick_labels
    else:
        major_tick_pos = _major_tick_positions(nfreqs)
        major_tick_labels = [freq_tick_labels[idx] for idx in major_tick_pos]

    totals = feature_importance_3d.sum(axis=2)
    global_ymax = float(np.max(totals)) if totals.size > 0 else 0.0
    use_log_gain = (importance_type == 'gain') and (gain_axis_scale == 'log')
    eps = 1e-12

    if use_log_gain:
        positive_totals = totals[totals > 0]

        if positive_totals.size > 0:
            auto_ymin = max(float(np.min(positive_totals)) * 0.8, eps)
            auto_ymax = max(float(np.max(positive_totals)) * 1.1, auto_ymin * 10.0)
        else:
            auto_ymin = eps
            auto_ymax = 1.0
    else:
        auto_ymin = 0.0
        auto_ymax = max(global_ymax * 1.05, 1e-9)

    global_ymin, global_ymax = _resolve_y_limits(y_limits_cfg, auto_ymin, auto_ymax, use_log_gain)

    for igroup in range(ngroups):
        row = igroup // ncols
        col = igroup % ncols
        ax = axes[row][col]

        mat = feature_importance_3d[igroup, :, :]

        # np.full() creates an array filled with specified constant value
        # (eps in case of a log scale)
        bottoms = np.full(nfreqs, eps, dtype=float) if use_log_gain else np.zeros(nfreqs, dtype=float)

        for iparm, parm_name in enumerate(parm_labels):
            vals = mat[:, iparm]

            # np.clip() clips the data to interval specified by 2nd and 3d arg
            plot_vals = np.clip(vals, eps, None) if use_log_gain else vals
            ax.bar(x, plot_vals, bottom=bottoms, color=palette[iparm], width=0.85, label=parm_name)
            bottoms += plot_vals

        ax.set_title(group_labels[igroup])
        ax.set_xticks(major_tick_pos)

        if row in rows_with_freq_ticks:
            ax.set_xticklabels(major_tick_labels, rotation=0, ha='center')
            ax.set_xlabel('Frequency')
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        if use_log_gain:
            ax.set_yscale('log')

        ax.set_ylim(global_ymin, global_ymax)

        if col == 0:
            ax.set_ylabel(f'{importance_type} (log)' if use_log_gain else importance_type)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

    for i in range(ngroups, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row][col].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc='upper right', frameon=True, title='Parameter')

    fig.suptitle(
        f'Feature importance ({importance_type}) for target label: {target_label} '
        f'(ignore_confidence={ignore_confidence})'
    )
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])

    outfname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfname, dpi=150, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f'Feature-importance plot saved to: {outfname}')

