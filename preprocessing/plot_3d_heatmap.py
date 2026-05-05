"""General-purpose 3D heatmap + median subplot plotting utilities."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import seaborn as sns


def plot_3d_heatmap(
    data_3d,
    dim1_labels=None,
    dim2_labels=None,
    dim3_labels=None,
    outfname=None,
    suptitle=None,
    cfg=None,
):
    """
    Plot a 3D array as a row of panels, one per fixed index of dimension 3.

    Each panel contains a heatmap for ``data_3d[:, :, i3]`` and a line plot of
    median values across the first dimension as a function of the second
    dimension.

    **NOTE**. While this is a generic function, historically the following data was used
    when developing the code: the 3D array is an array of classifier feature importances;
    first dimension is ROIs (channel names), 2nd dimension is frequencies, and 3d dimension
    is stat parameter used. This may help understanding the code.

    Args:
        data_3d (numpy.ndarray): Input array with shape ``(dim1, dim2, dim3)``.
        dim1_labels (Sequence[object] | None): Labels/values for dimension 1
            (heatmap y-axis). Defaults to 0-based row indices.
        dim2_labels (Sequence[object] | None): Labels/values for dimension 2
            (heatmap/median x-axis). Defaults to 0-based column indices.
        dim3_labels (Sequence[object] | None): Labels/values for dimension 3
            (panel titles). Defaults to 0-based parameter indices.
        outfname (str | pathlib.Path | None): Optional output file path.
        suptitle (str | None): Optional figure-level title.
        cfg (dict | None): Optional plotting configuration dictionary. Supported
            keys include:

            - ``font_scale`` (float, default ``1.0``)
            - ``style`` (str, default ``'whitegrid'``)
            - ``figure_size`` (list/tuple[width, height] or ``None``)
            - ``panel_height_ratios`` (list/tuple[heatmap_ratio, median_ratio],
              default ``[3.0, 1.2]``)
            - ``axis_scale`` (str, ``'linear'`` or ``'log'``)
            - ``heatmap_limits`` (list/tuple/dict or ``None``)
            - ``color_scale`` (str, colormap name)
            - ``hspace`` (float), ``wspace`` (float)
            - ``median_row_mask`` (array-like[bool] or ``None``)
            - ``median_y_limits`` (list/tuple[ymin, ymax] or ``None``)
            - ``heatmap_y_axis_font_size`` (float or ``None``)
            - ``x_axis_label`` (str), ``y_axis_label`` (str), ``median_y_label`` (str)
            - ``colorbar_label`` (object)
            - ``x_tick_mode`` (str: ``'auto'``, ``'round_frequency'``, ``'major'``)
            - ``dpi`` (int)
            - ``show_plots`` (bool)
            - ``median_exclude_zeros`` (bool)

    Returns:
        matplotlib.figure.Figure: Created figure.

    """
    cfg = cfg if isinstance(cfg, dict) else {}
    vals = np.asarray(data_3d, dtype=float)

    if vals.ndim != 3:
        raise ValueError(f'plot_3d_heatmap expects a 3D array, found shape={vals.shape}')

    nrows, ncols, npanels = vals.shape
    y_labels = _normalize_axis_labels(dim1_labels, nrows, 'dim1_labels')
    x_labels = _normalize_axis_labels(dim2_labels, ncols, 'dim2_labels')
    panel_labels = _normalize_axis_labels(dim3_labels, npanels, 'dim3_labels')

    font_scale = float(cfg.get('font_scale', 1.0))

    if font_scale <= 0:
        raise ValueError('plot_3d_heatmap cfg.font_scale must be > 0')

    style = str(cfg.get('style', 'whitegrid'))
    sns.set_theme(style=style, font_scale=font_scale)

    figure_size_cfg = cfg.get('figure_size', None)
    fig_w, fig_h = _resolve_3d_heatmap_figure_size(figure_size_cfg, npanels, nrows)

    panel_height_ratios_cfg = cfg.get('panel_height_ratios', [3.0, 1.2])

    if not isinstance(panel_height_ratios_cfg, (list, tuple)) or len(panel_height_ratios_cfg) != 2:
        raise ValueError('plot_3d_heatmap cfg.panel_height_ratios must be [heatmap_ratio, median_ratio]')

    heat_ratio = float(panel_height_ratios_cfg[0])
    median_ratio = float(panel_height_ratios_cfg[1])

    if heat_ratio <= 0 or median_ratio <= 0:
        raise ValueError('plot_3d_heatmap cfg.panel_height_ratios values must be > 0')

    axis_scale = str(cfg.get('axis_scale', cfg.get('gain_axis_scale', 'linear'))).lower()

    if axis_scale not in ('linear', 'log'):
        raise ValueError('plot_3d_heatmap cfg.axis_scale must be "linear" or "log"')

    limits_cfg = cfg.get('heatmap_limits', cfg.get('limits', cfg.get('color_limits', None)))
    cmap_name = str(cfg.get('color_scale', cfg.get('colormap', cfg.get('cmap_name', 'mako'))))

    norms = _resolve_heatmap_norm_by_panel(
        feature_importance_3d=vals,
        parm_labels=panel_labels,
        limits_cfg=limits_cfg,
        gain_axis_scale=axis_scale,
    )

    hspace = float(cfg.get('hspace', 0.15))
    wspace = float(cfg.get('wspace', 0.35))
    fig = plt.figure(figsize=(fig_w, fig_h), layout='constrained')
    gs = fig.add_gridspec(
        nrows=2,
        ncols=npanels,
        height_ratios=[heat_ratio, median_ratio],
        hspace=hspace,
        wspace=wspace,
    )

    median_row_mask = cfg.get('median_row_mask', None)
    median_y_limits_cfg = cfg.get('median_y_limits', None)
    heatmap_y_axis_font_size_cfg = cfg.get('heatmap_y_axis_font_size', None)

    if heatmap_y_axis_font_size_cfg is None:
        heatmap_y_axis_font_size = None
    else:
        heatmap_y_axis_font_size = float(heatmap_y_axis_font_size_cfg)

        if heatmap_y_axis_font_size <= 0:
            raise ValueError('plot_3d_heatmap cfg.heatmap_y_axis_font_size must be > 0')

    x_axis_label = str(cfg.get('x_axis_label', 'Column'))
    y_axis_label = str(cfg.get('y_axis_label', 'Row'))
    median_y_label = str(cfg.get('median_y_label', 'Median across rows'))
    colorbar_label = cfg.get('colorbar_label', f'value ({axis_scale})')
    x_tick_mode = str(cfg.get('x_tick_mode', 'auto')).lower()
    median_exclude_zeros = bool(cfg.get('median_exclude_zeros', True))

    for ipanel, panel_name in enumerate(panel_labels):
        ax_hm = fig.add_subplot(gs[0, ipanel])
        ax_md = fig.add_subplot(gs[1, ipanel], sharex=ax_hm)
        _plot_heatmap_median_panel(
            ax_heat=ax_hm,
            ax_median=ax_md,
            heatmap_2d=vals[:, :, ipanel],
            roi_labels=y_labels,
            freq_labels=x_labels,
            parm_name=panel_name,
            importance_type='value',
            gain_axis_scale=axis_scale,
            cmap_name=cmap_name,
            norm=norms[panel_name],
            median_row_mask=median_row_mask,
            median_y_limits_cfg=median_y_limits_cfg,
            heatmap_y_axis_font_size=heatmap_y_axis_font_size,
            show_roi_labels=(ipanel == 0),
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            median_y_label=median_y_label,
            colorbar_label=str(colorbar_label),
            x_tick_mode=x_tick_mode,
            median_exclude_zeros=median_exclude_zeros,
        )

    if suptitle is not None:
        fig.suptitle(str(suptitle))

    dpi = int(cfg.get('dpi', 300))

    if dpi <= 0:
        raise ValueError('plot_3d_heatmap cfg.dpi must be > 0')

    if outfname is not None:
        out_path = Path(outfname)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')

    if bool(cfg.get('show_plots', True)):
        plt.show()
    else:
        plt.close(fig)

    return fig


def _normalize_axis_labels(labels, size, arg_name):
    """Normalize optional axis labels to a list of strings of expected length."""
    if labels is None:
        return [str(i) for i in range(size)]

    if not isinstance(labels, (list, tuple, np.ndarray)):
        raise ValueError(f'plot_3d_heatmap {arg_name} must be a sequence or None')

    if len(labels) != size:
        raise ValueError(
            f'plot_3d_heatmap {arg_name} length mismatch: '
            f'expected {size}, got {len(labels)}'
        )

    return [str(item) for item in labels]


def _resolve_3d_heatmap_figure_size(figure_size_cfg, nparms, nchans):
    """
    Resolve figure size for 3D heatmap/median panel plots.

    Args:
        figure_size_cfg (list[float] | tuple[float, float] | None): Optional
            explicit figure size.
        nparms (int): Number of parameter panels.
        nchans (int): Number of ROI rows in each heatmap.

    Returns:
        tuple[float, float]: Figure width and height in inches.

    """
    if figure_size_cfg is not None:
        if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
            raise ValueError('plot_3d_heatmap cfg.figure_size must be null or [width, height]')

        width = float(figure_size_cfg[0])
        height = float(figure_size_cfg[1])

        if width <= 0 or height <= 0:
            raise ValueError('plot_3d_heatmap cfg.figure_size values must be > 0')

        return width, height

    per_panel_w = 3.5 + 0.04 * min(nchans, 120)
    width = max(8.0, per_panel_w * max(1, int(nparms)))
    height = max(4.0, 2.3 + 0.055 * min(nchans, 120))
    return width, height


def _resolve_heatmap_limits(limits_value, auto_min, auto_max, use_log_scale):
    """
    Resolve configured or automatic color limits for one heatmap.

    Args:
        limits_value (Sequence[float | None] | None): Optional [vmin, vmax].
        auto_min (float): Automatically inferred lower bound.
        auto_max (float): Automatically inferred upper bound.
        use_log_scale (bool): Whether log-normalized color scale is used.

    Returns:
        tuple[float, float]: Validated (vmin, vmax).

    """
    if limits_value is None:
        vmin, vmax = auto_min, auto_max
    else:
        if not isinstance(limits_value, (list, tuple)) or len(limits_value) != 2:
            raise ValueError('plot_3d_heatmap cfg.limits must be null, [vmin, vmax], or dict of those')

        vmin_cfg, vmax_cfg = limits_value
        vmin = auto_min if vmin_cfg is None else float(vmin_cfg)
        vmax = auto_max if vmax_cfg is None else float(vmax_cfg)

    if use_log_scale:
        if vmin <= 0 or vmax <= 0:
            raise ValueError('For log color scale, both limits must be > 0')

    if vmax <= vmin:
        raise ValueError(f'Invalid color limits: vmax ({vmax}) must be greater than vmin ({vmin})')

    return vmin, vmax


def _resolve_heatmap_norm_by_panel(feature_importance_3d, parm_labels, limits_cfg, gain_axis_scale):
    """
    Build per-panel color normalization objects for 3D heatmaps.

    Args:
        feature_importance_3d (numpy.ndarray): Importance values with shape
            (nchans, nfreqs, nparms).
        parm_labels (Sequence[str]): Ordered parameter labels.
        limits_cfg (list | tuple | dict | None): Shared or per-parameter limits.
        gain_axis_scale (str): Requested color mapping scale, linear or log.

    Returns:
        dict[str, matplotlib.colors.Normalize]: Color norm per parameter label.

    """
    use_log_scale = (gain_axis_scale == 'log')
    eps = 1e-12

    if limits_cfg is not None and not isinstance(limits_cfg, dict):
        shared_limits = limits_cfg
        per_parm_limits = {}
    elif isinstance(limits_cfg, dict):
        shared_limits = None
        per_parm_limits = limits_cfg
    else:
        shared_limits = None
        per_parm_limits = {}

    vals = np.asarray(feature_importance_3d, dtype=float)

    if use_log_scale:
        positive = vals[vals > 0]

        if positive.size == 0:
            auto_global_min = eps
            auto_global_max = 1.0
        else:
            auto_global_min = max(float(np.min(positive)), eps)
            auto_global_max = max(float(np.max(positive)), auto_global_min * 10.0)
    else:
        auto_global_min = 0.0
        auto_global_max = max(float(np.max(vals)), 1e-12)

    norms = {}

    for iparm, parm_name in enumerate(parm_labels):
        parm_vals = vals[:, :, iparm]

        if shared_limits is None and parm_name in per_parm_limits:
            limits_value = per_parm_limits[parm_name]

            if use_log_scale:
                parm_pos = parm_vals[parm_vals > 0]

                if parm_pos.size > 0:
                    auto_min = max(float(np.min(parm_pos)), eps)
                    auto_max = max(float(np.max(parm_pos)), auto_min * 10.0)
                else:
                    auto_min = eps
                    auto_max = 1.0
            else:
                auto_min = 0.0
                auto_max = max(float(np.max(parm_vals)), 1e-12)
        else:
            limits_value = shared_limits
            auto_min = auto_global_min
            auto_max = auto_global_max

        vmin, vmax = _resolve_heatmap_limits(limits_value, auto_min, auto_max, use_log_scale)

        if use_log_scale:
            norms[parm_name] = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norms[parm_name] = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return norms


def _plot_heatmap_median_panel(
    ax_heat,
    ax_median,
    heatmap_2d,
    roi_labels,
    freq_labels,
    parm_name,
    importance_type,
    gain_axis_scale,
    cmap_name,
    norm,
    median_row_mask=None,
    median_y_limits_cfg=None,
    heatmap_y_axis_font_size=None,
    show_roi_labels=True,
    x_axis_label='Frequency (Hz)',
    y_axis_label='ROI',
    median_y_label='Median across ROI',
    colorbar_label=None,
    x_tick_mode='auto',
    median_exclude_zeros=True,
):
    """
    Render one heatmap/median panel for a fixed third-dimension slice.

    Args:
        ax_heat (matplotlib.axes.Axes): Axis for the heatmap.
        ax_median (matplotlib.axes.Axes): Axis for median-vs-frequency trace.
        heatmap_2d (numpy.ndarray): Importance matrix with shape (nchans, nfreqs).
        roi_labels (Sequence[str]): ROI labels in display order (y-axis).
        freq_labels (Sequence[str]): Frequency labels in display order.
        parm_name (str): Parameter name shown in panel title.
        importance_type (str): Importance metric name for labels.
        gain_axis_scale (str): Color mapping scale, linear or log.
        cmap_name (str): Matplotlib colormap name.
        norm (matplotlib.colors.Normalize): Pre-resolved color normalization.
        median_row_mask (array-like[bool] | None): Optional boolean mask over
            heatmap rows indicating which ROI rows should contribute to the
            median trace. If None, all rows are used.
        median_y_limits_cfg (list[float | None] | tuple[float | None, float | None] | None):
            Optional explicit y-axis limits for the median panel.
        heatmap_y_axis_font_size (float | None): Optional explicit font size for
            heatmap y-axis label text (tick labels and axis label).
        median_exclude_zeros (bool): When ``True``, exclude zero values from the
            median calculation. When ``False``, use all rows/values as-is.

    Returns:
        None

    """
    mat = np.asarray(heatmap_2d, dtype=float)
    use_log_scale = (gain_axis_scale == 'log')
    eps = 1e-12

    display_mat = np.clip(mat, eps, None) if use_log_scale else mat

    im = ax_heat.imshow(
        display_mat,
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        cmap=cmap_name,
        norm=norm,
    )
    ax_heat.grid(False)

    nfreqs = display_mat.shape[1]

    if x_tick_mode == 'round_frequency':
        round_tick_pos, round_tick_labels = _round_frequency_ticks(freq_labels)
    elif x_tick_mode == 'major':
        round_tick_pos, round_tick_labels = None, None
    else:
        round_tick_pos, round_tick_labels = _round_frequency_ticks(freq_labels)

    if round_tick_pos is not None:
        major_tick_pos = round_tick_pos
    else:
        major_tick_pos = _major_tick_positions(nfreqs)

    ax_heat.set_xticks(major_tick_pos)

    if round_tick_pos is not None:
        ax_heat.set_xticklabels(round_tick_labels, rotation=0)
    else:
        ax_heat.set_xticklabels(
            [_format_frequency_labels(freq_labels)[idx] for idx in major_tick_pos],
            rotation=0,
        )

    ax_heat.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    ax_heat.set_yticks(np.arange(len(roi_labels), dtype=int))

    if show_roi_labels:
        ax_heat.set_yticklabels(roi_labels)
        if heatmap_y_axis_font_size is None:
            ax_heat.set_ylabel(y_axis_label)
        else:
            ax_heat.set_ylabel(y_axis_label, fontsize=heatmap_y_axis_font_size)
            ax_heat.tick_params(axis='y', labelsize=heatmap_y_axis_font_size)
    else:
        ax_heat.set_yticklabels([])
        ax_heat.set_ylabel('')

    ax_heat.set_title(f'{parm_name}')

    cbar = plt.colorbar(im, ax=ax_heat, pad=0.01)

    if colorbar_label is None:
        cbar.set_label(f'{importance_type} ({gain_axis_scale})')
    else:
        cbar.set_label(colorbar_label)

    if median_row_mask is None:
        median_source = mat
    else:
        row_mask = np.asarray(median_row_mask, dtype=bool)

        if row_mask.shape[0] != mat.shape[0]:
            raise ValueError('median_row_mask length must match number of heatmap rows')

        if np.any(row_mask):
            median_source = mat[row_mask, :]
        else:
            # Fallback for pathological configuration; keep legacy behavior.
            median_source = mat

    if median_exclude_zeros:
        # All purely zero values in each frequency column correspond to (channel, frequency)
        # pairs that have no importance. Exclude those from the median calculations.
        nonzero_median_source = np.where(median_source != 0.0, median_source, np.nan)
        valid_cols = np.any(~np.isnan(nonzero_median_source), axis=0)
        median_vals = np.zeros(nonzero_median_source.shape[1], dtype=float)

        if np.any(valid_cols):
            median_vals[valid_cols] = np.nanmedian(nonzero_median_source[:, valid_cols], axis=0)
    else:
        # Use all values including zeros for median calculation.
        median_vals = np.median(median_source, axis=0)

    median_plot = np.clip(median_vals, eps, None) if use_log_scale else median_vals
    ax_median.plot(np.arange(nfreqs), median_plot, color='black', linewidth=1.6)
    ax_median.set_xlim(ax_heat.get_xlim())
    ax_median.set_xticks(major_tick_pos)

    if round_tick_pos is not None:
        ax_median.set_xticklabels(round_tick_labels)
    else:
        ax_median.set_xticklabels([_format_frequency_labels(freq_labels)[idx] for idx in major_tick_pos])

    ax_median.set_xlabel(x_axis_label)
    ax_median.set_ylabel(median_y_label)
    ax_median.grid(True, axis='y', alpha=0.3)

    if use_log_scale:
        ax_median.set_yscale('log')

    auto_ymin, auto_ymax = ax_median.get_ylim()
    ymin, ymax = _resolve_y_limits(median_y_limits_cfg, auto_ymin, auto_ymax, use_log_scale)
    ax_median.set_ylim(ymin, ymax)


def _format_frequency_labels(freq_labels):
    """
    Convert labels like '8.00Hz' or '8Hz' to numeric-only strings.

    Args:
        freq_labels (Sequence[object]): Frequency labels that may include units
            or other surrounding text.

    Returns:
        list[str]: Frequency labels converted to plain numeric strings where
        possible, otherwise the original label text.

    """
    out = []

    for freq in freq_labels:
        txt = str(freq).strip()
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)

        if match is None:
            out.append(txt)
            continue

        val = float(match.group(0))
        out.append(f'{val:g}')

    return out


def _major_tick_positions(nfreqs, max_major_ticks=6):
    """
    Return index positions for major x-axis ticks.

    Args:
        nfreqs (int): Number of frequency bins available on the x-axis.
        max_major_ticks (int): Maximum number of major ticks to place.

    Returns:
        numpy.ndarray: Integer tick positions spanning the available frequency
        indices.

    """
    if nfreqs <= max_major_ticks:
        return np.arange(nfreqs, dtype=int)

    pos = np.linspace(0, nfreqs - 1, num=max_major_ticks)
    pos = np.unique(np.round(pos).astype(int))
    pos[0] = 0
    pos[-1] = nfreqs - 1
    return pos


def _round_frequency_ticks(freq_labels):
    """
    Select tick positions closest to target frequencies 1, 2, 4, 8, 16, 32 Hz.

    Args:
        freq_labels (Sequence[object]): Frequency labels to parse into numeric
            values.

    Returns:
        tuple[numpy.ndarray | None, list[str] | None]: Tick positions and tick
        labels for round-number frequencies, or ``(None, None)`` when the input
        labels cannot be parsed.

    """
    target_freqs = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)

    parsed = []

    for txt in freq_labels:
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(txt))

        if match is None:
            return None, None

        parsed.append(float(match.group(0)))

    freqs = np.asarray(parsed, dtype=float)

    if freqs.size == 0:
        return None, None

    chosen = []

    for target in target_freqs:
        idx = int(np.argmin(np.abs(freqs - target)))
        chosen.append((idx, int(target)))

    # Deduplicate if several targets map to the same bin, keep left-to-right order.
    dedup = {}

    for idx, target_int in chosen:
        if idx not in dedup:
            dedup[idx] = target_int

    pos = np.array(sorted(dedup.keys()), dtype=int)
    labels = [f'{dedup[idx]}' for idx in pos]

    return pos.astype(int), labels


def _resolve_y_limits(y_limits_cfg, auto_ymin, auto_ymax, use_log_gain):
    """
    Resolve manual/automatic y-axis limits.

    Args:
        y_limits_cfg (list[float | None] | tuple[float | None, float | None] | None):
            User-specified y-axis limits, or ``None`` for automatic limits.
        auto_ymin (float): Automatically computed lower limit.
        auto_ymax (float): Automatically computed upper limit.
        use_log_gain (bool): Whether the y-axis will use logarithmic scaling.

    Returns:
        tuple[float, float]: Final ``(ymin, ymax)`` pair for the plot.

    Notes:
        ``y_limits_cfg`` can be ``None`` to keep automatic limits, or a
        two-element sequence ``[ymin, ymax]`` where either entry may be
        ``None`` to fall back to the automatic value.

    """
    if y_limits_cfg is None:
        return auto_ymin, auto_ymax

    if not isinstance(y_limits_cfg, (list, tuple)) or len(y_limits_cfg) != 2:
        raise ValueError('plot_3d_heatmap cfg.y_limits must be null or [ymin, ymax]')

    ymin_cfg, ymax_cfg = y_limits_cfg
    ymin = auto_ymin if ymin_cfg is None else float(ymin_cfg)
    ymax = auto_ymax if ymax_cfg is None else float(ymax_cfg)

    if use_log_gain:
        if ymin <= 0 or ymax <= 0:
            raise ValueError('For log gain axis, both y_limits must be > 0')

    if ymax <= ymin:
        raise ValueError(f'Invalid y_limits: ymax ({ymax}) must be greater than ymin ({ymin})')

    return ymin, ymax
