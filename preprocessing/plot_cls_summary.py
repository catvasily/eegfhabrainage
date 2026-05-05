"""
Create classification summary plots from summary CSV files.
"""
from pathlib import Path
import colorsys

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import numpy as np
import pandas as pd
import seaborn as sns

STEP = 'summary_plots'
_AUC4METHOD_ORDER = [
    'Abnormality',
    'Focal Epi',
    'Focal Non-epi',
    'Gen Epi',
    'Gen Non-epi',
]
_AUC4PHYSICIAN_ORDER = [
    'All',
    'Sophia',
    'Maria',
    'Eleni',
    'Zoe',
    'Athina',
    'Others',
]


def plot_cls_summary(ss):
    """
    Entry point for classifier summary plotting step.

    Expected JSON structure::

        "summary_plots": {
            "what": "auc4method",
            "drop_rows_on_empty_col": "AUC",
            "auc4method": {
                "input_csv": "classification_results_summary.csv"
            },
            "auc4physician": {
                "input_csv": "classification_results_summary.csv",
                "dim_reduction": ["to_lobes"]
            }
        }

    Args:
        ss(obj): reference to run_classifier application object

    Returns:
        Nothing

    """
    cfg = ss.args.get('summary_plots', None)

    if not isinstance(cfg, dict):
        raise ValueError('Missing or invalid "summary_plots" configuration in classifier input JSON.')

    what = str(cfg.get('what', '')).strip()
    if not what:
        raise ValueError('summary_plots.what must be a non-empty string.')

    what_cfg = cfg.get(what, None)
    if not isinstance(what_cfg, dict):
        raise ValueError(f'summary_plots.{what} must be a dictionary.')

    input_csv = what_cfg.get('input_csv', None)
    if input_csv is None:
        raise ValueError(f'summary_plots.{what}.input_csv is required.')

    csv_path = _resolve_input_csv_path(input_csv, ss.out_root)
    if not csv_path.exists():
        raise FileNotFoundError(f'Input CSV does not exist: {csv_path}')

    df = pd.read_csv(csv_path)

    drop_col = cfg.get('drop_rows_on_empty_col', None)
    if drop_col is not None:
        drop_col = str(drop_col).strip()
        if drop_col:
            if drop_col not in df.columns:
                raise ValueError(
                    f'summary_plots.drop_rows_on_empty_col="{drop_col}", '
                    f'but this column is missing in input CSV: {csv_path}'
                )
            before_rows = len(df)
            df = _drop_rows_on_empty_column(df, drop_col)
            removed = before_rows - len(df)
            print(f'Rows removed due to empty "{drop_col}": {removed}')

    if what == 'auc4method':
        _plot_auc4method(ss, cfg, what_cfg, df, csv_path)
    elif what == 'auc4physician':
        _plot_auc4physician(ss, cfg, what_cfg, df, csv_path)
    else:
        raise ValueError(
            f'Unknown summary_plots.what="{what}". '
            'Currently implemented values: ["auc4method", "auc4physician"].'
        )

    print(f'\n Step *{STEP}* completed successfully')


def _resolve_input_csv_path(input_csv, out_root):
    csv_path = Path(str(input_csv))
    if not csv_path.is_absolute():
        csv_path = Path(out_root) / csv_path
    return csv_path


def _resolve_output_plot_path(output_png, out_root, what):
    if output_png is None:
        return Path(out_root) / f'cls_summary_{what}.png'

    plot_path = Path(str(output_png))
    if not plot_path.is_absolute():
        plot_path = Path(out_root) / plot_path
    return plot_path


def _drop_rows_on_empty_column(df, col_name):
    values = df[col_name]

    keep_mask = values.notna() & (values.astype(str).str.strip() != '')
    return df.loc[keep_mask].copy()


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in ('1', 'true', 't', 'yes', 'y'):
        return True
    if text in ('0', 'false', 'f', 'no', 'n', ''):
        return False

    return False


def _sorted_dim_values(values):
    unique_vals = []
    for val in values:
        sval = str(val).strip()
        if sval and sval not in unique_vals:
            unique_vals.append(sval)
    return sorted(unique_vals)


def _lighten_color(color, amount=0.45):
    """
    Return a paler version of an RGB/hex color by increasing lightness.
    """
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1.0 - amount * (1.0 - l)
    rp, gp, bp = colorsys.hls_to_rgb(h, l, s)
    return to_hex((rp, gp, bp))


def _get_legend_max_rows(auc_cfg, cfg_path, default=3):
    max_rows = auc_cfg.get('legend_max_rows', default)
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{cfg_path}.legend_max_rows must be an integer >= 1.') from exc

    if max_rows < 1:
        raise ValueError(f'{cfg_path}.legend_max_rows must be >= 1.')

    return max_rows


def _get_legend_ncol(n_items, max_rows):
    n_items = int(n_items)
    if n_items <= 0:
        return 1
    return max(1, int(np.ceil(n_items / float(max_rows))))


def _get_legend_bbox_to_anchor(auc_cfg, cfg_path):
    bbox = auc_cfg.get('legend_bbox_to_anchor', [0.5, 1.0])
    if bbox is None:
        return None

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 2:
        raise ValueError(f'{cfg_path}.legend_bbox_to_anchor must be null or [x, y].')

    return (float(bbox[0]), float(bbox[1]))


def _plot_auc4method(ss, cfg, auc_cfg, df, csv_path):
    required_cols = ['Label', 'AUC', 'dim_reduction', 'ignore_confidence']
    missing = [name for name in required_cols if name not in df.columns]
    if missing:
        raise ValueError(
            f'Input CSV for auc4method is missing required columns: {missing}. '
            f'CSV: {csv_path}'
        )

    wrk = df.copy()
    wrk['Label'] = wrk['Label'].astype(str).str.strip()
    wrk['dim_reduction'] = wrk['dim_reduction'].astype(str).str.strip()
    wrk['ignore_confidence'] = wrk['ignore_confidence'].map(_to_bool)
    wrk['AUC'] = pd.to_numeric(wrk['AUC'], errors='coerce')

    wrk = wrk[wrk['Label'].isin(_AUC4METHOD_ORDER)].copy()
    wrk = wrk[wrk['dim_reduction'] != ''].copy()
    wrk = wrk[wrk['AUC'].notna()].copy()

    if wrk.empty:
        raise ValueError('No rows left to plot for auc4method after filtering and conversion.')

    # Average repeated rows for the same (Label, dim_reduction, ignore_confidence) tuple.
    grp = (
        wrk.groupby(['Label', 'dim_reduction', 'ignore_confidence'], as_index=False)['AUC']
        .mean()
    )

    dim_values = _sorted_dim_values(grp['dim_reduction'])

    # Build hue order in grouped pairs: dim_reduction with ignore_confidence=False then True.
    hue_order = []
    for dim in dim_values:
        for ign in (False, True):
            if ((grp['dim_reduction'] == dim) & (grp['ignore_confidence'] == ign)).any():
                hue_order.append(_combo_key(dim, ign))

    if not hue_order:
        raise ValueError('No valid [dim_reduction, ignore_confidence] combinations found for auc4method.')

    grp['combo'] = grp.apply(lambda row: _combo_key(row['dim_reduction'], row['ignore_confidence']), axis=1)
    grp['Label'] = pd.Categorical(grp['Label'], categories=_AUC4METHOD_ORDER, ordered=True)

    base_colors = sns.color_palette('tab10', n_colors=max(1, len(dim_values)))
    dim_to_base = {dim: base_colors[idx] for idx, dim in enumerate(dim_values)}

    palette = {}
    for dim in dim_values:
        base = dim_to_base[dim]
        palette[_combo_key(dim, False)] = to_hex(base)
        palette[_combo_key(dim, True)] = _lighten_color(base, amount=0.45)

    fig_size = auc_cfg.get('figure_size', [13, 6])
    if not isinstance(fig_size, (list, tuple)) or len(fig_size) != 2:
        raise ValueError('summary_plots.auc4method.figure_size must be [width, height].')

    fig_w = float(fig_size[0])
    fig_h = float(fig_size[1])
    if fig_w <= 0 or fig_h <= 0:
        raise ValueError('summary_plots.auc4method.figure_size values must be > 0.')

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_width = float(auc_cfg.get('bar_width', 0.16))
    pair_gap = float(auc_cfg.get('pair_gap', 0.08))

    if bar_width <= 0:
        raise ValueError('summary_plots.auc4method.bar_width must be > 0.')
    if pair_gap < 0:
        raise ValueError('summary_plots.auc4method.pair_gap must be >= 0.')

    x_centers = np.arange(len(_AUC4METHOD_ORDER), dtype=float)
    bars_per_pair = 2
    total_group_width = len(dim_values) * bars_per_pair * bar_width + max(0, len(dim_values) - 1) * pair_gap
    left_edge = -0.5 * total_group_width

    grp_lookup = {
        (str(row['Label']), str(row['dim_reduction']), bool(row['ignore_confidence'])): float(row['AUC'])
        for _, row in grp.iterrows()
    }

    for idim, dim in enumerate(dim_values):
        pair_start = left_edge + idim * (bars_per_pair * bar_width + pair_gap)
        false_center = pair_start + 0.5 * bar_width
        true_center = pair_start + 1.5 * bar_width

        false_heights = [grp_lookup.get((label, dim, False), np.nan) for label in _AUC4METHOD_ORDER]
        true_heights = [grp_lookup.get((label, dim, True), np.nan) for label in _AUC4METHOD_ORDER]

        false_label = _combo_key(dim, False)
        true_label = _combo_key(dim, True)

        ax.bar(
            x_centers + false_center,
            false_heights,
            width=bar_width,
            color=palette[false_label],
            label=false_label,
            align='center',
        )
        ax.bar(
            x_centers + true_center,
            true_heights,
            width=bar_width,
            color=palette[true_label],
            label=true_label,
            align='center',
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(_AUC4METHOD_ORDER)

    title = auc_cfg.get('title', 'AUC by Label for dim_reduction and ignore_confidence combinations')
    ax.set_title(str(title))
    ax.set_xlabel('Label')
    ax.set_ylabel('AUC')

    y_limits = auc_cfg.get('y_limits', None)
    if y_limits is not None:
        if not isinstance(y_limits, (list, tuple)) or len(y_limits) != 2:
            raise ValueError('summary_plots.auc4method.y_limits must be null or [ymin, ymax].')
        ymin, ymax = y_limits
        ymin = None if ymin is None else float(ymin)
        ymax = None if ymax is None else float(ymax)
        ax.set_ylim(bottom=ymin, top=ymax)

    legend_title = auc_cfg.get('legend_title', '[dim_reduction, ignore_confidence]')
    legend_font_size = float(auc_cfg.get('legend_font_size', 9))
    legend_title_font_size = float(auc_cfg.get('legend_title_font_size', 10))

    if legend_font_size <= 0 or legend_title_font_size <= 0:
        raise ValueError('summary_plots.auc4method legend font sizes must be > 0.')

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    legend_max_rows = _get_legend_max_rows(auc_cfg, 'summary_plots.auc4method', default=3)
    legend_ncol = _get_legend_ncol(len(uniq_labels), legend_max_rows)
    legend_loc = str(auc_cfg.get('legend_loc', 'upper center'))
    legend_bbox_to_anchor = _get_legend_bbox_to_anchor(auc_cfg, 'summary_plots.auc4method')

    ax.legend(
        uniq_handles,
        uniq_labels,
        title=str(legend_title),
        frameon=True,
        fontsize=legend_font_size,
        title_fontsize=legend_title_font_size,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
    )

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha('right')

    plt.tight_layout()

    out_png = _resolve_output_plot_path(auc_cfg.get('output_png', None), ss.out_root, 'auc4method')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=int(auc_cfg.get('dpi', 150)), bbox_inches='tight')

    if ss.args.get('show_plots', True):
        plt.show()
    else:
        plt.close()

    print(f'auc4method plot saved to: {out_png}')


def _plot_auc4physician(ss, cfg, auc_cfg, df, csv_path):
    required_cols = ['Label', 'AUC', 'dim_reduction', 'Physician', 'ignore_confidence']
    missing = [name for name in required_cols if name not in df.columns]
    if missing:
        raise ValueError(
            f'Input CSV for auc4physician is missing required columns: {missing}. '
            f'CSV: {csv_path}'
        )

    dim_reduction_list = auc_cfg.get('dim_reduction', None)
    if not isinstance(dim_reduction_list, list) or not dim_reduction_list:
        raise ValueError(
            'summary_plots.auc4physician.dim_reduction must be a non-empty list of strings.'
        )

    dim_reduction_values = []
    for val in dim_reduction_list:
        sval = str(val).strip()
        if sval and sval not in dim_reduction_values:
            dim_reduction_values.append(sval)
    if not dim_reduction_values:
        raise ValueError(
            'summary_plots.auc4physician.dim_reduction must contain at least one non-empty string.'
        )

    wrk = df.copy()
    wrk['Label'] = wrk['Label'].astype(str).str.strip()
    wrk['dim_reduction'] = wrk['dim_reduction'].astype(str).str.strip()
    wrk['AUC'] = pd.to_numeric(wrk['AUC'], errors='coerce')
    wrk['Physician'] = wrk['Physician'].fillna('').astype(str).str.strip()
    wrk['ignore_confidence'] = wrk['ignore_confidence'].map(_to_bool)
    wrk.loc[wrk['Physician'] == '', 'Physician'] = 'All'

    wrk = wrk[wrk['Label'].isin(_AUC4METHOD_ORDER)].copy()
    wrk = wrk[wrk['dim_reduction'].isin(dim_reduction_values)].copy()
    wrk = wrk[wrk['AUC'].notna()].copy()

    if wrk.empty:
        raise ValueError('No rows left to plot for auc4physician after filtering and conversion.')

    # Average repeated rows for the same (Label, dim_reduction, Physician, ignore_confidence) tuple.
    grp = (
        wrk.groupby(['Label', 'dim_reduction', 'Physician', 'ignore_confidence'], as_index=False)['AUC']
        .mean()
    )

    physicians_present = set(grp['Physician'].astype(str).tolist())
    physician_order = [name for name in _AUC4PHYSICIAN_ORDER if name in physicians_present]
    extras = sorted(name for name in physicians_present if name not in set(_AUC4PHYSICIAN_ORDER))
    physician_order.extend(extras)

    if not physician_order:
        raise ValueError('No physician values found for auc4physician after preprocessing.')

    fig_size = auc_cfg.get('figure_size', [13, max(4.0, 3.6 * len(dim_reduction_values))])
    if not isinstance(fig_size, (list, tuple)) or len(fig_size) != 2:
        raise ValueError('summary_plots.auc4physician.figure_size must be [width, height].')

    fig_w = float(fig_size[0])
    fig_h = float(fig_size[1])
    if fig_w <= 0 or fig_h <= 0:
        raise ValueError('summary_plots.auc4physician.figure_size values must be > 0.')

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(
        nrows=len(dim_reduction_values),
        ncols=1,
        figsize=(fig_w, fig_h),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    group_gap = float(auc_cfg.get('group_gap', 0.03))

    bar_width_cfg = auc_cfg.get('bar_width', None)
    pair_gap = float(auc_cfg.get('pair_gap', 0.03))
    if bar_width_cfg is None:
        # Auto-size bars so physician groups remain readable as physician count grows.
        auto_width = (
            0.9 - max(0, len(physician_order) - 1) * group_gap - len(physician_order) * pair_gap
        ) / max(1, 2 * len(physician_order))
        bar_width = max(0.03, min(0.18, auto_width))
    else:
        bar_width = float(bar_width_cfg)

    bars_per_pair = 2
    bars_per_group = len(physician_order) * bars_per_pair

    if bar_width <= 0:
        raise ValueError('summary_plots.auc4physician.bar_width must be > 0.')
    if group_gap < 0:
        raise ValueError('summary_plots.auc4physician.group_gap must be >= 0.')
    if pair_gap < 0:
        raise ValueError('summary_plots.auc4physician.pair_gap must be >= 0.')
    if (
        bars_per_group * bar_width
        + max(0, len(physician_order) - 1) * (group_gap + pair_gap)
    ) <= 0:
        raise ValueError('summary_plots.auc4physician bar geometry is invalid; check bar_width/group_gap.')

    palette_colors = sns.color_palette('tab20', n_colors=max(1, len(physician_order)))
    physician_palette = {}
    for idx, physician in enumerate(physician_order):
        base_color = to_hex(palette_colors[idx])
        physician_palette[(physician, False)] = base_color
        physician_palette[(physician, True)] = _lighten_color(base_color, amount=0.45)

    grp_lookup = {
        (
            str(row['Label']),
            str(row['dim_reduction']),
            str(row['Physician']),
            bool(row['ignore_confidence']),
        ): float(row['AUC'])
        for _, row in grp.iterrows()
    }

    x_centers = np.arange(len(_AUC4METHOD_ORDER), dtype=float)
    total_group_width = (
        bars_per_group * bar_width
        + max(0, len(physician_order) - 1) * (group_gap + pair_gap)
    )
    left_edge = -0.5 * total_group_width

    for iax, (ax, dim) in enumerate(zip(axes, dim_reduction_values)):
        for idx, physician in enumerate(physician_order):
            pair_start = left_edge + idx * (bars_per_pair * bar_width + pair_gap + group_gap)
            false_center = pair_start + 0.5 * bar_width
            true_center = pair_start + 1.5 * bar_width

            heights_false = [
                grp_lookup.get((label, dim, physician, False), np.nan)
                for label in _AUC4METHOD_ORDER
            ]
            heights_true = [
                grp_lookup.get((label, dim, physician, True), np.nan)
                for label in _AUC4METHOD_ORDER
            ]

            ax.bar(
                x_centers + false_center,
                heights_false,
                width=bar_width,
                color=physician_palette[(physician, False)],
                label=f'{physician} | ignore_confidence=False',
                align='center',
            )
            ax.bar(
                x_centers + true_center,
                heights_true,
                width=bar_width,
                color=physician_palette[(physician, True)],
                label=f'{physician} | ignore_confidence=True',
                align='center',
            )

        ax.set_title(f'dim_reduction: {dim}')
        ax.set_ylabel('AUC')

        y_limits = auc_cfg.get('y_limits', None)
        if y_limits is not None:
            if not isinstance(y_limits, (list, tuple)) or len(y_limits) != 2:
                raise ValueError('summary_plots.auc4physician.y_limits must be null or [ymin, ymax].')
            ymin, ymax = y_limits
            ymin = None if ymin is None else float(ymin)
            ymax = None if ymax is None else float(ymax)
            ax.set_ylim(bottom=ymin, top=ymax)

        legend_title = auc_cfg.get('legend_title', 'Physician | ignore_confidence')
        legend_font_size = float(auc_cfg.get('legend_font_size', 9))
        legend_title_font_size = float(auc_cfg.get('legend_title_font_size', 10))
        if legend_font_size <= 0 or legend_title_font_size <= 0:
            raise ValueError('summary_plots.auc4physician legend font sizes must be > 0.')

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            uniq_handles.append(handle)
            uniq_labels.append(label)

        # When plotting multiple dim_reduction panels, show the legend only once.
        if len(dim_reduction_values) == 1 or iax == 0:
            legend_max_rows = _get_legend_max_rows(auc_cfg, 'summary_plots.auc4physician', default=3)
            legend_ncol = _get_legend_ncol(len(uniq_labels), legend_max_rows)
            legend_loc = str(auc_cfg.get('legend_loc', 'upper center'))
            legend_bbox_to_anchor = _get_legend_bbox_to_anchor(auc_cfg, 'summary_plots.auc4physician')

            ax.legend(
                uniq_handles,
                uniq_labels,
                title=str(legend_title),
                frameon=True,
                fontsize=legend_font_size,
                title_fontsize=legend_title_font_size,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=legend_ncol,
            )

    axes[-1].set_xticks(x_centers)
    axes[-1].set_xticklabels(_AUC4METHOD_ORDER)
    axes[-1].set_xlabel('Label')

    for tick in axes[-1].get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha('right')

    overall_title = auc_cfg.get('title', 'AUC by Label grouped by Physician for fixed dim_reduction')
    fig.suptitle(str(overall_title))
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = _resolve_output_plot_path(auc_cfg.get('output_png', None), ss.out_root, 'auc4physician')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=int(auc_cfg.get('dpi', 150)), bbox_inches='tight')

    if ss.args.get('show_plots', True):
        plt.show()
    else:
        plt.close()

    print(f'auc4physician plot saved to: {out_png}')


def _combo_key(dim_reduction, ignore_confidence):
    return f'{dim_reduction} | ignore_confidence={bool(ignore_confidence)}'
