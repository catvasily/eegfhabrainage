"""Plot utilities for cls_predict summarize outputs."""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, precision_recall_curve

from cls_predict import (
    _build_predict_summarize_csv_path,
    _build_summarize_model_cfg,
    _normalize_physician_list,
    _resolve_predict_calibration_method,
)
from auc_bootstrap_stats import pairwise_error_mcc


def plot_predict_outputs(ss, predict_cfg):
    """Create all predict plot-only outputs and return their saved paths."""
    ridge_png = plot_predict_ridge_grid(ss, predict_cfg)
    counts_png = plot_predict_counts_bar_grid(ss, predict_cfg)
    aucs_png = plot_predict_aucs_bar_grid(ss, predict_cfg)
    grid_pr_png = plot_predict_grid_pr(ss, predict_cfg)
    mcc_png = plot_predict_mcc_error_heatmap(ss, predict_cfg)
    return {
        'ridge_plot': ridge_png,
        'counts_bar_plot': counts_png,
        'aucs_bar_plot': aucs_png,
        'grid_pr_plot': grid_pr_png,
        'mcc_error_plot': mcc_png,
    }


def plot_predict_mcc_error_heatmap(ss, predict_cfg):
    """Build MCC heatmap of pairwise physician error vectors from summarize CSV."""
    mcc_cfg = predict_cfg.get('mcc_error_plot', {}) or {}

    if not bool(mcc_cfg.get('enabled', True)):
        return None

    resolved = _resolve_predict_plot_inputs(ss, predict_cfg)
    summarize_cfg = resolved['summarize_cfg']
    model_cfg = resolved['model_cfg']
    model_physicians = resolved['model_physicians']
    summary_csv_path = resolved['summary_csv_path']
    df_plot = resolved['df_plot']

    calc_df = pd.DataFrame(index=df_plot.index)
    calc_df['TrueLabel'] = pd.to_numeric(df_plot.get('TrueLabel', np.nan), errors='coerce')

    for physician_name in model_physicians:
        if physician_name not in df_plot.columns:
            raise ValueError(
                f'Missing expected physician predicted-label column in summary CSV: {physician_name}'
            )

        calc_df[physician_name] = pd.to_numeric(df_plot[physician_name], errors='coerce')

    valid_mask = calc_df['TrueLabel'].isin([0, 1])

    for physician_name in model_physicians:
        valid_mask &= calc_df[physician_name].isin([0, 1])

    calc_df = calc_df.loc[valid_mask].copy()

    if calc_df.empty:
        raise ValueError(
            'No rows with valid binary labels were found for MCC error heatmap computation.'
        )

    y_true = calc_df['TrueLabel'].to_numpy(dtype=int)
    y_pred = calc_df[model_physicians].to_numpy(dtype=int)
    mcc_matrix = pairwise_error_mcc(y_true=y_true, y_pred=y_pred)

    output_png_cfg = mcc_cfg.get('output_png', None)

    if output_png_cfg is None or not str(output_png_cfg).strip():
        png_path = summary_csv_path.with_name(f'{summary_csv_path.stem}_mcc_error_heatmap.png')
    else:
        output_path = Path(str(output_png_cfg).strip())

        if output_path.is_absolute():
            png_path = output_path
        else:
            png_path = summary_csv_path.parent / output_path

    _draw_mcc_error_heatmap(
        mcc_matrix=mcc_matrix,
        physician_labels=model_physicians,
        out_png=png_path,
        target_label=str(summarize_cfg.get('target_label', model_cfg.get('target_label', ''))),
        mcc_cfg=mcc_cfg,
    )
    return png_path


def _draw_mcc_error_heatmap(
    mcc_matrix,
    physician_labels,
    out_png,
    target_label,
    mcc_cfg,
):
    """Render and save pairwise physician error-MCC heatmap."""
    style = str(mcc_cfg.get('style', 'white')).strip() or 'white'
    context = str(mcc_cfg.get('context', 'paper')).strip() or 'paper'
    sns.set_theme(style=style, context=context)

    figure_size = mcc_cfg.get('figure_size', [8.0, 6.8])

    if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
        raise ValueError('predict.mcc_error_plot.figure_size must be [width, height].')

    figure_size = (float(figure_size[0]), float(figure_size[1]))
    dpi = int(mcc_cfg.get('dpi', 300))

    cmap = str(mcc_cfg.get('cmap', 'RdBu_r')).strip() or 'RdBu_r'
    annot = bool(mcc_cfg.get('annot', True))
    annot_fmt = str(mcc_cfg.get('fmt', '.2f')).strip() or '.2f'
    linewidths = float(mcc_cfg.get('linewidths', 0.5))
    linecolor = str(mcc_cfg.get('linecolor', 'white')).strip() or 'white'
    square = bool(mcc_cfg.get('square', True))

    vmin_cfg = mcc_cfg.get('vmin', -1.0)
    vmax_cfg = mcc_cfg.get('vmax', 1.0)
    center_cfg = mcc_cfg.get('center', 0.0)

    vmin = None if vmin_cfg is None else float(vmin_cfg)
    vmax = None if vmax_cfg is None else float(vmax_cfg)
    center = None if center_cfg is None else float(center_cfg)

    tick_font_size = float(mcc_cfg.get('tick_font_size', 9.0))
    label_font_size = float(mcc_cfg.get('label_font_size', 10.0))
    title_font_size = float(mcc_cfg.get('title_font_size', 12.0))
    title_y = float(mcc_cfg.get('title_y', 0.995))

    x_label = str(mcc_cfg.get('x_label', 'Physician model')).strip() or 'Physician model'
    y_label = str(mcc_cfg.get('y_label', 'Physician model')).strip() or 'Physician model'

    cbar = bool(mcc_cfg.get('cbar', True))
    cbar_label = str(mcc_cfg.get('cbar_label', 'MCC of error vectors')).strip()

    title_cfg = mcc_cfg.get('title', None)

    if title_cfg is None or not str(title_cfg).strip():
        label = str(target_label).strip()

        if label:
            fig_title = f'Pairwise Error MCC (target: {label})'
        else:
            fig_title = 'Pairwise Error MCC'
    else:
        fig_title = str(title_cfg)

    fig, ax = plt.subplots(figsize=figure_size)

    cbar_kws = {'label': cbar_label} if cbar and cbar_label else None

    sns.heatmap(
        mcc_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot=annot,
        fmt=annot_fmt,
        linewidths=linewidths,
        linecolor=linecolor,
        square=square,
        xticklabels=physician_labels,
        yticklabels=physician_labels,
        cbar=cbar,
        cbar_kws=cbar_kws,
        ax=ax,
    )

    ax.set_xlabel(x_label, fontsize=label_font_size)
    ax.set_ylabel(y_label, fontsize=label_font_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_font_size)

    fig.suptitle(fig_title, y=title_y, fontsize=title_font_size)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')

    if bool(mcc_cfg.get('show_plot', False)):
        plt.show()

    plt.close(fig)


def _resolve_predict_plot_inputs(ss, predict_cfg):
    """Resolve common predict-summary inputs used by plot-only visualizations."""
    summarize_cfg = predict_cfg.get('summarize', {}) or {}

    model_physicians = _normalize_physician_list(
        summarize_cfg.get('model_physicians', None),
        'predict.summarize.model_physicians',
    )

    if not model_physicians:
        raise ValueError('predict.summarize.model_physicians must contain at least one physician.')

    target_physicians = _normalize_physician_list(
        summarize_cfg.get('target_physicians', summarize_cfg.get('target_physicans', None)),
        'predict.summarize.target_physicians',
    )

    model_cfg = _build_summarize_model_cfg(ss, predict_cfg, summarize_cfg)
    threshold_selection = model_cfg.get('threshold_selection', None)

    summary_csv_path = _build_predict_summarize_csv_path(
        ss=ss,
        summarize_cfg=summarize_cfg,
        model_cfg=model_cfg,
        model_physicians=model_physicians,
        target_physicians=target_physicians,
        threshold_selection=threshold_selection,
        calibration_method=_resolve_predict_calibration_method(predict_cfg),
    )

    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f'Predict summary CSV file for plotting does not exist: {summary_csv_path}'
        )

    df = pd.read_csv(summary_csv_path)
    df_plot = _prepare_plot_dataframe(df)

    if df_plot.empty:
        raise ValueError(
            f'No valid rows with TrueLabel and Confident in {{0,1}} were found in {summary_csv_path}'
        )

    joint_method = summarize_cfg['joint_prediction'].get('method', 'joint_LR')

    return {
        'summarize_cfg': summarize_cfg,
        'model_cfg': model_cfg,
        'model_physicians': model_physicians,
        'summary_csv_path': summary_csv_path,
        'df_summary': df,
        'df_plot': df_plot,
        'joint_method': joint_method,
    }


def plot_predict_ridge_grid(ss, predict_cfg):
    """
    Build a physician grid where each cell contains 4 stacked ridge plots of
    confidence distributions by (TrueLabel, Confident) subset.

    Returns:
        Path to saved PNG file, or None when disabled via config.
    """
    ridge_cfg = predict_cfg.get('ridge_plots', {}) or {}

    if not bool(ridge_cfg.get('enabled', True)):
        return None

    resolved = _resolve_predict_plot_inputs(ss, predict_cfg)
    summarize_cfg = resolved['summarize_cfg']
    model_cfg = resolved['model_cfg']
    model_physicians = resolved['model_physicians']
    summary_csv_path = resolved['summary_csv_path']
    df_plot = resolved['df_plot']
    joint_method = resolved['joint_method']

    png_path = summary_csv_path.with_suffix('.png')
    _draw_physician_ridge_grid(
        df_plot=df_plot,
        model_physicians=model_physicians,
        out_png=png_path,
        csv_name=summary_csv_path.name,
        target_label=str(summarize_cfg.get('target_label', model_cfg.get('target_label', ''))),
        model_ignore_confidence=bool(model_cfg.get('ignore_confidence', False)),
        ridge_cfg=ridge_cfg,
        joint_method = joint_method
    )
    return png_path


def plot_predict_counts_bar_grid(ss, predict_cfg):
    """
    Build a 4x2 grid of count bar plots from predict-summary outputs.

    Left column: physician + joint counts by subset row.
    Right column: STD across physician counts by subset row.

    Returns:
        Path to saved PNG file, or None when disabled via config.
    """
    counts_cfg = predict_cfg.get('counts_bar_plot', {}) or {}

    if not bool(counts_cfg.get('enabled', True)):
        return None

    ridge_cfg = predict_cfg.get('ridge_plots', {}) or {}
    resolved = _resolve_predict_plot_inputs(ss, predict_cfg)

    summarize_cfg = resolved['summarize_cfg']
    model_physicians = resolved['model_physicians']
    summary_csv_path = resolved['summary_csv_path']
    df_plot = resolved['df_plot']
    joint_method = resolved['joint_method']

    png_path = summary_csv_path.with_name(f'{summary_csv_path.stem}_counts_bar_plot.png')

    _draw_counts_bar_plot_grid(
        df_plot=df_plot,
        model_physicians=model_physicians,
        out_png=png_path,
        csv_name=summary_csv_path.name,
        target_label=str(summarize_cfg.get('target_label', '')),
        counts_cfg=counts_cfg,
        ridge_cfg=ridge_cfg,
        joint_method=joint_method,
    )
    return png_path


def plot_predict_aucs_bar_grid(ss, predict_cfg):
    """
    Build a 3x1 grid of PR AUC bars from summarize-only CSV outputs.

    Rows are rendered in this order:
    1) confident subset
    2) non-confident subset
    3) overall subset

    Returns:
        Path to saved PNG file, or None when disabled via config.
    """
    aucs_cfg = predict_cfg.get('aucs_bar_plot', {}) or {}

    if not bool(aucs_cfg.get('enabled', True)):
        return None

    resolved = _resolve_predict_plot_inputs(ss, predict_cfg)
    summarize_cfg = resolved['summarize_cfg']
    model_cfg = resolved['model_cfg']
    model_physicians = resolved['model_physicians']
    summary_csv_path = resolved['summary_csv_path']
    df_summary = resolved['df_summary']

    output_png_cfg = aucs_cfg.get('output_png', None)

    if output_png_cfg is None or not str(output_png_cfg).strip():
        png_path = summary_csv_path.with_name(f'{summary_csv_path.stem}_pr_auc_bar_plot.png')
    else:
        output_path = Path(str(output_png_cfg).strip())

        if output_path.is_absolute():
            png_path = output_path
        else:
            png_path = summary_csv_path.parent / output_path

    _draw_aucs_bar_plot_grid(
        df_summary=df_summary,
        model_physicians=model_physicians,
        out_png=png_path,
        target_label=str(summarize_cfg.get('target_label', model_cfg.get('target_label', ''))),
        aucs_cfg=aucs_cfg,
    )
    return png_path


def plot_predict_grid_pr(ss, predict_cfg):
    """
    Build a physician PR-curve grid from summarize-only predict CSV outputs.

    For each subset row and physician column, panel contains two PR curves:
    physician and joint model on the same subset.

    Returns:
        Path to saved PNG file, or None when disabled via config.
    """
    grid_cfg = predict_cfg.get('grid_pr_plot', {}) or {}

    if not bool(grid_cfg.get('enabled', False)):
        return None

    resolved = _resolve_predict_plot_inputs(ss, predict_cfg)
    summarize_cfg = resolved['summarize_cfg']
    model_cfg = resolved['model_cfg']
    model_physicians = resolved['model_physicians']
    summary_csv_path = resolved['summary_csv_path']
    df_summary = resolved['df_summary']

    output_png_cfg = grid_cfg.get('output_png', None)

    if output_png_cfg is None or not str(output_png_cfg).strip():
        png_path = summary_csv_path.with_name(f'{summary_csv_path.stem}_grid_pr_plot.png')
    else:
        output_path = Path(str(output_png_cfg).strip())

        if output_path.is_absolute():
            png_path = output_path
        else:
            png_path = summary_csv_path.parent / output_path

    _draw_grid_pr_plot(
        df_summary=df_summary,
        model_physicians=model_physicians,
        out_png=png_path,
        target_label=str(summarize_cfg.get('target_label', model_cfg.get('target_label', ''))),
        grid_cfg=grid_cfg,
    )
    return png_path


def _confidence_to_positive_probability(pred_series, confidence_series):
    """Convert confidence-of-predicted-class to positive-class probability."""
    pred_num = pd.to_numeric(pred_series, errors='coerce')
    conf_num = pd.to_numeric(confidence_series, errors='coerce')

    out = pd.Series(np.nan, index=pred_series.index, dtype=float)
    valid_mask = (
        pred_num.isin([0.0, 1.0])
        & np.isfinite(conf_num)
        & (conf_num >= 0.0)
        & (conf_num <= 1.0)
    )

    if valid_mask.any():
        pred_arr = pred_num.loc[valid_mask].to_numpy(dtype=int)
        conf_arr = conf_num.loc[valid_mask].to_numpy(dtype=float)
        out.loc[valid_mask] = np.where(pred_arr == 1, conf_arr, 1.0 - conf_arr)

    return out


def _compute_pr_curve_stats(y_true_series, y_score_series):
    """Compute PR curve arrays and AP for valid binary rows."""
    y_true_num = pd.to_numeric(y_true_series, errors='coerce')
    y_score_num = pd.to_numeric(y_score_series, errors='coerce')

    valid_mask = y_true_num.isin([0.0, 1.0]) & np.isfinite(y_score_num)

    if not valid_mask.any():
        return None

    y_true = y_true_num.loc[valid_mask].to_numpy(dtype=int)
    y_score = y_score_num.loc[valid_mask].to_numpy(dtype=float)

    if y_true.size < 2 or np.unique(y_true).size < 2:
        return {
            'recall': None,
            'precision': None,
            'ap': np.nan,
            'n': int(y_true.size),
        }

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    return {
        'recall': recall,
        'precision': precision,
        'ap': ap,
        'n': int(y_true.size),
    }


def _draw_grid_pr_plot(
    df_summary,
    model_physicians,
    out_png,
    target_label,
    grid_cfg,
):
    """Render and save subset-by-physician grid of PR curves."""
    style = str(grid_cfg.get('style', 'whitegrid')).strip() or 'whitegrid'
    context = str(grid_cfg.get('context', 'paper')).strip() or 'paper'
    sns.set_theme(style=style, context=context)

    required_base_cols = ['ScanID', 'TrueLabel', 'Confident', 'JointLabel', 'JointConfidence', 'held_out_set']

    missing_base_cols = [col for col in required_base_cols if col not in df_summary.columns]

    if missing_base_cols:
        raise ValueError(
            'predict.grid_pr_plot requires summary CSV columns: '
            f'{required_base_cols}. Missing: {missing_base_cols}'
        )

    physician_conf_cols = [f'{physician_name}_PredConfidence' for physician_name in model_physicians]
    required_phys_cols = list(model_physicians) + physician_conf_cols
    missing_phys_cols = [col for col in required_phys_cols if col not in df_summary.columns]

    if missing_phys_cols:
        raise ValueError(
            'predict.grid_pr_plot is missing physician columns in summary CSV: '
            f'{missing_phys_cols}'
        )

    plot_df = df_summary.copy()
    scan_id_text = plot_df['ScanID'].astype(str).str.strip()

    # Keep only scan-level records and drop explicit summary rows like brier_scores_(...).
    valid_scan_mask = scan_id_text.ne('') & (~scan_id_text.str.startswith('brier_scores_', na=False))

    plot_df['TrueLabel'] = pd.to_numeric(plot_df['TrueLabel'], errors='coerce')
    plot_df['Confident'] = pd.to_numeric(plot_df['Confident'], errors='coerce')
    plot_df['held_out_set'] = pd.to_numeric(plot_df['held_out_set'], errors='coerce')

    binary_target_mask = plot_df['TrueLabel'].isin([0.0, 1.0]) & plot_df['Confident'].isin([0.0, 1.0])
    plot_df = plot_df.loc[valid_scan_mask & binary_target_mask].copy()

    if plot_df.empty:
        raise ValueError(
            'No eligible scan-level rows available for predict.grid_pr_plot '
            '(requires valid ScanID, TrueLabel in {0,1}, Confident in {0,1}).'
        )

    plot_df['TrueLabel'] = plot_df['TrueLabel'].astype(int)
    plot_df['Confident'] = plot_df['Confident'].astype(int)

    for physician_name in model_physicians:
        label_col = physician_name
        conf_col = f'{physician_name}_PredConfidence'
        plot_df[f'__P_{physician_name}'] = _confidence_to_positive_probability(
            plot_df[label_col],
            plot_df[conf_col],
        )

    plot_df['__P_Joint'] = _confidence_to_positive_probability(
        plot_df['JointLabel'],
        plot_df['JointConfidence'],
    )

    subset_defs = [
        {
            'key': 'confident',
            'label': str(grid_cfg.get('confident_label', 'Confident true labels')).strip()
            or 'Confident true labels',
            'mask': plot_df['Confident'] == 1,
        },
        {
            'key': 'nonconfident',
            'label': str(grid_cfg.get('nonconfident_label', 'Non-confident true labels')).strip()
            or 'Non-confident true labels',
            'mask': plot_df['Confident'] == 0,
        },
        {
            'key': 'all',
            'label': str(grid_cfg.get('all_label', 'All eligible records')).strip()
            or 'All eligible records',
            'mask': pd.Series(True, index=plot_df.index),
        },
    ]

    joint_eligible_mask = plot_df['held_out_set'] == 0

    joint_curves = {}

    for subset_def in subset_defs:
        subset_mask = subset_def['mask']
        joint_mask = subset_mask & joint_eligible_mask
        joint_curves[subset_def['key']] = _compute_pr_curve_stats(
            plot_df.loc[joint_mask, 'TrueLabel'],
            plot_df.loc[joint_mask, '__P_Joint'],
        )

    physician_curves = {subset_def['key']: {} for subset_def in subset_defs}

    for subset_def in subset_defs:
        subset_mask = subset_def['mask']

        for physician_name in model_physicians:
            physician_curves[subset_def['key']][physician_name] = _compute_pr_curve_stats(
                plot_df.loc[subset_mask, 'TrueLabel'],
                plot_df.loc[subset_mask, f'__P_{physician_name}'],
            )

    n_physicians = len(model_physicians)
    panel_rows = int(grid_cfg.get('n_panel_rows', 1))
    panel_cols = int(grid_cfg.get('n_panel_cols', 5))

    if panel_rows <= 0 or panel_cols <= 0:
        raise ValueError('predict.grid_pr_plot.n_panel_rows and n_panel_cols must be positive integers.')

    if panel_rows * panel_cols < n_physicians:
        raise ValueError(
            'predict.grid_pr_plot panel grid is too small for configured model physicians. '
            f'Need at least {n_physicians} cells, got {panel_rows * panel_cols}.'
        )

    figure_size_cfg = grid_cfg.get('figure_size', None)

    if figure_size_cfg is None:
        width_per_col = float(grid_cfg.get('width_per_col', 3.7))
        height_per_row = float(grid_cfg.get('height_per_subset_row', 2.8))
        figure_size = (
            panel_cols * width_per_col,
            len(subset_defs) * panel_rows * height_per_row,
        )
    else:
        if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
            raise ValueError('predict.grid_pr_plot.figure_size must be null or [width, height].')

        figure_size = (float(figure_size_cfg[0]), float(figure_size_cfg[1]))

    x_limits_cfg = grid_cfg.get('x_limits', [0.0, 1.0])
    y_limits_cfg = grid_cfg.get('y_limits', [0.0, 1.0])

    if not isinstance(x_limits_cfg, (list, tuple)) or len(x_limits_cfg) != 2:
        raise ValueError('predict.grid_pr_plot.x_limits must be [xmin, xmax].')

    if not isinstance(y_limits_cfg, (list, tuple)) or len(y_limits_cfg) != 2:
        raise ValueError('predict.grid_pr_plot.y_limits must be [ymin, ymax].')

    x_limits = (float(x_limits_cfg[0]), float(x_limits_cfg[1]))
    y_limits = (float(y_limits_cfg[0]), float(y_limits_cfg[1]))

    if x_limits[1] <= x_limits[0]:
        raise ValueError('predict.grid_pr_plot.x_limits must satisfy xmax > xmin.')

    if y_limits[1] <= y_limits[0]:
        raise ValueError('predict.grid_pr_plot.y_limits must satisfy ymax > ymin.')

    joint_color = str(grid_cfg.get('joint_curve_color', '#f58518')).strip() or '#f58518'
    physician_color = str(grid_cfg.get('physician_curve_color', '#4c78a8')).strip() or '#4c78a8'
    show_legend = bool(grid_cfg.get('show_legend', True))

    physician_tpl = str(
        grid_cfg.get(
            'physician_label_template',
            '{physician} (AP={ap:.3f}, n={n})',
        )
    )
    joint_tpl = str(grid_cfg.get('joint_label_template', 'Joint (AP={ap:.3f}, n={n})'))

    total_rows = len(subset_defs) * panel_rows
    fig, axes = plt.subplots(
        total_rows,
        panel_cols,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    no_curve_text = str(grid_cfg.get('no_curve_text', 'insufficient class variation')).strip()

    for subset_idx, subset_def in enumerate(subset_defs):
        base_row = subset_idx * panel_rows
        joint_stats = joint_curves[subset_def['key']]

        for physician_idx, physician_name in enumerate(model_physicians):
            row_in_block = physician_idx // panel_cols
            col_in_block = physician_idx % panel_cols
            ax = axes[base_row + row_in_block, col_in_block]

            phys_stats = physician_curves[subset_def['key']][physician_name]

            if phys_stats is not None and phys_stats['recall'] is not None:
                phys_label = physician_tpl.format(
                    physician=physician_name,
                    ap=float(phys_stats['ap']),
                    n=int(phys_stats['n']),
                )
                ax.plot(
                    phys_stats['recall'],
                    phys_stats['precision'],
                    color=physician_color,
                    linewidth=float(grid_cfg.get('line_width', 1.8)),
                    label=phys_label,
                )
            else:
                ax.text(
                    0.03,
                    0.08,
                    f'{physician_name}: {no_curve_text}',
                    transform=ax.transAxes,
                    fontsize=float(grid_cfg.get('note_font_size', 7.0)),
                    color=physician_color,
                    ha='left',
                    va='bottom',
                )

            if joint_stats is not None and joint_stats['recall'] is not None:
                joint_label = joint_tpl.format(
                    physician=physician_name,
                    ap=float(joint_stats['ap']),
                    n=int(joint_stats['n']),
                )
                ax.plot(
                    joint_stats['recall'],
                    joint_stats['precision'],
                    color=joint_color,
                    linewidth=float(grid_cfg.get('line_width', 1.8)),
                    linestyle=str(grid_cfg.get('joint_line_style', '--')).strip() or '--',
                    label=joint_label,
                )
            else:
                ax.text(
                    0.03,
                    0.02,
                    f'Joint: {no_curve_text}',
                    transform=ax.transAxes,
                    fontsize=float(grid_cfg.get('note_font_size', 7.0)),
                    color=joint_color,
                    ha='left',
                    va='bottom',
                )

            if subset_idx == 0:
                ax.set_title(
                    physician_name,
                    fontsize=float(grid_cfg.get('panel_title_font_size', 10.0)),
                )

            if col_in_block == 0:
                ax.set_ylabel(
                    f"{subset_def['label']}\n"
                    + (str(grid_cfg.get('y_label', 'Precision')).strip() or 'Precision')
                )
            else:
                ax.set_ylabel('')

            if row_in_block == (panel_rows - 1):
                ax.set_xlabel(str(grid_cfg.get('x_label', 'Recall')).strip() or 'Recall')
            else:
                ax.set_xlabel('')

            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.grid(alpha=float(grid_cfg.get('grid_alpha', 0.3)))

            if show_legend:
                ax.legend(
                    loc=str(grid_cfg.get('legend_loc', 'lower left')).strip() or 'lower left',
                    fontsize=float(grid_cfg.get('legend_font_size', 7.0)),
                    frameon=bool(grid_cfg.get('legend_frame', False)),
                )

        for idx_unused in range(n_physicians, panel_rows * panel_cols):
            row_in_block = idx_unused // panel_cols
            col_in_block = idx_unused % panel_cols
            axes[base_row + row_in_block, col_in_block].axis('off')

    title_cfg = grid_cfg.get('title', None)

    if title_cfg is None or not str(title_cfg).strip():
        label = str(target_label).strip()

        if label:
            title = f'PR curves by physician and subset (target: {label})'
        else:
            title = 'PR curves by physician and subset'
    else:
        title = str(title_cfg)

    fig.suptitle(
        title,
        y=float(grid_cfg.get('title_y', 0.995)),
        fontsize=float(grid_cfg.get('title_font_size', 12.0)),
    )
    fig.subplots_adjust(
        hspace=float(grid_cfg.get('hspace', 0.35)),
        wspace=float(grid_cfg.get('wspace', 0.2)),
    )

    fig.savefig(out_png, dpi=int(grid_cfg.get('dpi', 300)), bbox_inches='tight')

    if bool(grid_cfg.get('show_plot', False)):
        plt.show()

    plt.close(fig)


def _parse_mean_ci_text(value):
    """Parse strings like `0.812 (0.754, 0.864)` and return (mean, lo, hi)."""
    text = str(value).strip()

    if not text:
        return (np.nan, np.nan, np.nan)

    ci_match = re.match(
        r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\(\s*'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\s*$',
        text,
    )

    if ci_match:
        mean_val = float(ci_match.group(1))
        lo_val = float(ci_match.group(2))
        hi_val = float(ci_match.group(3))
        return (mean_val, lo_val, hi_val)

    scalar = pd.to_numeric(pd.Series([text]), errors='coerce').iloc[0]

    if pd.isna(scalar):
        return (np.nan, np.nan, np.nan)

    scalar = float(scalar)
    return (scalar, np.nan, np.nan)


def _extract_pr_auc_values_for_row(df_summary, row_name, model_physicians, joint_auc_col):
    """Extract ordered PR AUC values (mean/CI) for physician bars plus joint bar."""
    if 'ScanID' not in df_summary.columns:
        raise ValueError('Summary CSV is missing required column: ScanID')

    scan_id_norm = df_summary['ScanID'].astype(str).str.strip()
    row_name_norm = str(row_name).strip()
    row_matches = df_summary.loc[scan_id_norm == row_name_norm]

    if row_matches.empty:
        available_summary_rows = sorted(
            {
                val
                for val in scan_id_norm.tolist()
                if val.startswith('PR_AUC')
                or val.startswith('F1_')
                or val.startswith('joint_')
            }
        )
        raise ValueError(
            f'Required summary row {row_name!r} was not found in summarize CSV. '
            f'Available summary-like rows include: {available_summary_rows}'
        )

    row = row_matches.iloc[0]
    group_names = list(model_physicians) + ['Joint']
    means = []
    ci_lowers = []
    ci_uppers = []

    for physician_name in model_physicians:
        col_name = f'{physician_name}_PredConfidence'

        if col_name not in df_summary.columns:
            raise ValueError(
                f'Missing expected physician confidence summary column in CSV: {col_name}'
            )

        mean_val, lo_val, hi_val = _parse_mean_ci_text(row.get(col_name, np.nan))
        means.append(mean_val)
        ci_lowers.append(lo_val)
        ci_uppers.append(hi_val)

    if joint_auc_col not in df_summary.columns:
        raise ValueError(
            f'Missing expected joint summary column in CSV: {joint_auc_col}'
        )

    mean_val, lo_val, hi_val = _parse_mean_ci_text(row.get(joint_auc_col, np.nan))
    means.append(mean_val)
    ci_lowers.append(lo_val)
    ci_uppers.append(hi_val)

    return {
        'group_names': group_names,
        'means': np.asarray(means, dtype=float),
        'ci_lowers': np.asarray(ci_lowers, dtype=float),
        'ci_uppers': np.asarray(ci_uppers, dtype=float),
    }


def _draw_aucs_bar_plot_grid(
    df_summary,
    model_physicians,
    out_png,
    target_label,
    aucs_cfg,
):
    """Render and save vertical PR AUC bar plots for confident/non-confident/overall subsets."""
    style = str(aucs_cfg.get('style', 'whitegrid')).strip() or 'whitegrid'
    context = str(aucs_cfg.get('context', 'paper')).strip() or 'paper'
    sns.set_theme(style=style, context=context)

    show_ci = bool(aucs_cfg.get('show_ci', True))
    show_value_labels = bool(aucs_cfg.get('show_value_labels', True))
    value_label_fmt = str(aucs_cfg.get('value_label_format', '{:.3f}'))
    value_label_font_size = float(aucs_cfg.get('value_label_font_size', 8.0))
    figure_size = aucs_cfg.get('figure_size', [11.0, 12.0])

    if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
        raise ValueError('predict.aucs_bar_plot.figure_size must be [width, height].')

    figure_size = (float(figure_size[0]), float(figure_size[1]))
    dpi = int(aucs_cfg.get('dpi', 300))
    bar_width = float(aucs_cfg.get('bar_width', 0.72))
    y_limits_cfg = aucs_cfg.get('y_limits', [0.0, 1.0])

    if y_limits_cfg is not None:
        if not isinstance(y_limits_cfg, (list, tuple)) or len(y_limits_cfg) != 2:
            raise ValueError('predict.aucs_bar_plot.y_limits must be null or [ymin, ymax].')
        y_limits = (float(y_limits_cfg[0]), float(y_limits_cfg[1]))
        if y_limits[1] <= y_limits[0]:
            raise ValueError('predict.aucs_bar_plot.y_limits must satisfy ymax > ymin.')
    else:
        y_limits = None

    ci_capsize = float(aucs_cfg.get('ci_capsize', 4.0))
    ci_line_width = float(aucs_cfg.get('ci_line_width', 1.1))
    pane_title_font_size = float(aucs_cfg.get('pane_title_font_size', 10.0))
    title_font_size = float(aucs_cfg.get('title_font_size', 12.0))
    title_y = float(aucs_cfg.get('title_y', 0.995))
    grid_alpha = float(aucs_cfg.get('grid_alpha', 0.25))
    grid_hspace = float(aucs_cfg.get('grid_hspace', 0.28))

    joint_auc_col = str(aucs_cfg.get('joint_auc_column', 'JointConfidence')).strip() or 'JointConfidence'
    joint_label = str(aucs_cfg.get('joint_group_label', 'Joint')).strip() or 'Joint'

    colors_cfg = aucs_cfg.get('colors', None)

    if isinstance(colors_cfg, (list, tuple)) and len(colors_cfg) >= (len(model_physicians) + 1):
        bar_colors = [str(c) for c in colors_cfg[: (len(model_physicians) + 1)]]
    else:
        physician_color = str(aucs_cfg.get('physician_color', '#4c78a8')).strip() or '#4c78a8'
        joint_color = str(aucs_cfg.get('joint_color', '#f58518')).strip() or '#f58518'
        bar_colors = [physician_color for _ in model_physicians] + [joint_color]

    missing_color = str(aucs_cfg.get('missing_color', '#c7c7c7')).strip() or '#c7c7c7'

    subset_rows_cfg = aucs_cfg.get('subset_rows', {}) or {}
    subset_defs = [
        {
            'row_name': str(subset_rows_cfg.get('confident', 'PR_AUC_confident')),
            'label': str(aucs_cfg.get('confident_label', 'Confident labels')).strip() or 'Confident labels',
        },
        {
            'row_name': str(subset_rows_cfg.get('nonconfident', 'PR_AUC_nonconfident')),
            'label': str(aucs_cfg.get('nonconfident_label', 'Non-confident labels')).strip() or 'Non-confident labels',
        },
        {
            'row_name': str(subset_rows_cfg.get('overall', 'PR_AUC')),
            'label': str(aucs_cfg.get('overall_label', 'Overall')).strip() or 'Overall',
        },
    ]

    fig, axes = plt.subplots(3, 1, figsize=figure_size, sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for i_row, subset_def in enumerate(subset_defs):
        ax = axes[i_row]
        extracted = _extract_pr_auc_values_for_row(
            df_summary=df_summary,
            row_name=subset_def['row_name'],
            model_physicians=model_physicians,
            joint_auc_col=joint_auc_col,
        )

        group_names = list(extracted['group_names'])
        group_names[-1] = joint_label

        means = extracted['means']
        ci_lowers = extracted['ci_lowers']
        ci_uppers = extracted['ci_uppers']

        x = np.arange(len(group_names), dtype=float)
        missing_mask = ~np.isfinite(means)
        y_vals = np.where(np.isfinite(means), means, 0.0)

        plot_colors = list(bar_colors)

        for idx_missing, is_missing in enumerate(missing_mask):
            if is_missing:
                plot_colors[idx_missing] = missing_color

        yerr = None

        if show_ci:
            lo_err = means - ci_lowers
            hi_err = ci_uppers - means
            valid_ci = (
                np.isfinite(means)
                & np.isfinite(ci_lowers)
                & np.isfinite(ci_uppers)
                & (lo_err >= 0.0)
                & (hi_err >= 0.0)
            )

            if np.any(valid_ci):
                lo_plot = np.where(valid_ci, lo_err, 0.0)
                hi_plot = np.where(valid_ci, hi_err, 0.0)
                yerr = np.vstack([lo_plot, hi_plot])

        bars = ax.bar(
            x,
            y_vals,
            width=bar_width,
            color=plot_colors,
            edgecolor='black',
            linewidth=0.7,
            yerr=yerr,
            capsize=ci_capsize if yerr is not None else 0.0,
            error_kw={'elinewidth': ci_line_width, 'capthick': ci_line_width},
        )

        for idx_missing, is_missing in enumerate(missing_mask):
            if is_missing:
                bars[idx_missing].set_hatch('//')

        if show_value_labels:
            y_span = 1.0 if y_limits is None else (y_limits[1] - y_limits[0])
            y_offset = 0.015 * y_span

            for bar, mean_val, is_missing in zip(bars, means, missing_mask):
                if is_missing:
                    label_text = 'NA'
                    y_text = bar.get_height() + y_offset
                else:
                    label_text = value_label_fmt.format(float(mean_val))
                    y_text = bar.get_height() + y_offset

                ax.text(
                    bar.get_x() + 0.5 * bar.get_width(),
                    y_text,
                    label_text,
                    ha='center',
                    va='bottom',
                    fontsize=value_label_font_size,
                )

        ax.set_title(subset_def['label'], fontsize=pane_title_font_size)
        ax.set_ylabel(str(aucs_cfg.get('y_label', 'PR AUC')).strip() or 'PR AUC')
        ax.grid(axis='y', alpha=grid_alpha)

        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.axhline(0.0, color='black', linewidth=0.7, alpha=0.45)

    axes[-1].set_xticks(np.arange(len(model_physicians) + 1, dtype=float))
    axes[-1].set_xticklabels(list(model_physicians) + [joint_label])
    axes[-1].set_xlabel(
        str(aucs_cfg.get('x_label', 'Physicians and Joint model')).strip()
        or 'Physicians and Joint model'
    )

    title_cfg = aucs_cfg.get('title', None)
    title_label = str(target_label).strip()

    if title_cfg is None or not str(title_cfg).strip():
        if title_label:
            fig_title = f'PR AUC by predictor for target label: {title_label}'
        else:
            fig_title = 'PR AUC by predictor'
    else:
        fig_title = str(title_cfg)

    fig.suptitle(fig_title, y=title_y, fontsize=title_font_size)
    fig.subplots_adjust(hspace=grid_hspace)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')

    if bool(aucs_cfg.get('show_plot', False)):
        plt.show()

    plt.close(fig)


def _prepare_plot_dataframe(df):
    """Keep only rows needed for ridge plotting and normalize numeric columns."""
    out = df.copy()
    out['TrueLabel'] = pd.to_numeric(out.get('TrueLabel', np.nan), errors='coerce')
    out['Confident'] = pd.to_numeric(out.get('Confident', np.nan), errors='coerce')

    mask = out['TrueLabel'].isin([0, 1]) & out['Confident'].isin([0, 1])
    out = out.loc[mask].copy()

    if not out.empty:
        out['TrueLabel'] = out['TrueLabel'].astype(int)
        out['Confident'] = out['Confident'].astype(int)

    return out


def _draw_counts_bar_plot_grid(
    df_plot,
    model_physicians,
    out_png,
    csv_name,
    target_label,
    counts_cfg,
    ridge_cfg,
    joint_method,
):
    """Render and save counts bar plot grid for predict summary outputs."""
    style = str(counts_cfg.get('style', ridge_cfg.get('style', 'whitegrid'))).strip() or 'whitegrid'
    context = str(counts_cfg.get('context', ridge_cfg.get('context', 'paper'))).strip() or 'paper'
    sns.set_theme(style=style, context=context)

    colors = counts_cfg.get('colors', ridge_cfg.get('colors', ['#1a9850', '#91cf60', '#fc8d59', '#d73027']))

    if not isinstance(colors, (list, tuple)) or len(colors) < 4:
        raise ValueError('predict.counts_bar_plot.colors must contain at least 4 colors.')

    subset_labels_cfg = counts_cfg.get(
        'subset_labels',
        ridge_cfg.get('subset_labels', {}) or {},
    ) or {}
    subset_defs = [
        {
            'true_label': 0,
            'confident': 1,
            'label': subset_labels_cfg.get('0_1', 'Conf/N'),
            'overall': False,
        },
        {
            'true_label': 0,
            'confident': 0,
            'label': subset_labels_cfg.get('0_0', 'Non-conf/N'),
            'overall': False,
        },
        {
            'true_label': 1,
            'confident': 0,
            'label': subset_labels_cfg.get('1_0', 'Non-conf/A'),
            'overall': False,
        },
        {
            'true_label': 1,
            'confident': 1,
            'label': subset_labels_cfg.get('1_1', 'Conf/A'),
            'overall': False,
        },
        {
            'true_label': None,
            'confident': None,
            'label': subset_labels_cfg.get('overall', 'Overall'),
            'overall': True,
        },
    ]
    subset_index_by_pair = {
        (subset_def['true_label'], subset_def['confident']): idx
        for idx, subset_def in enumerate(subset_defs)
        if not subset_def['overall']
    }
    n_rows = len(subset_defs)

    method_token = 'LR' if joint_method == 'joint_LR' else ('Bayes' if joint_method == 'bayes' else '??')
    default_joint_label = f'Joint ({method_token})'
    joint_group_label = str(counts_cfg.get('joint_group_label', default_joint_label)).strip() or default_joint_label

    figure_size = counts_cfg.get('figure_size', None)

    if figure_size is None:
        width = float(counts_cfg.get('width', 13.0))
        height = float(counts_cfg.get('height', 12.0))
        figure_size = (width, height)
    else:
        if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
            raise ValueError('predict.counts_bar_plot.figure_size must be null or [width, height].')

        figure_size = (float(figure_size[0]), float(figure_size[1]))

    dpi = int(counts_cfg.get('dpi', 300))
    bar_width = float(counts_cfg.get('bar_width', 0.72))
    std_bar_width = float(counts_cfg.get('std_bar_width', bar_width))
    std_panel_x_limits = counts_cfg.get('std_panel_x_limits', [-0.75, 0.75])
    left_right_width_ratio = counts_cfg.get('column_width_ratio', [6.0, 1.8])

    if not isinstance(std_panel_x_limits, (list, tuple)) or len(std_panel_x_limits) != 2:
        raise ValueError(
            'predict.counts_bar_plot.std_panel_x_limits must be [xmin, xmax].'
        )

    std_panel_x_limits = (float(std_panel_x_limits[0]), float(std_panel_x_limits[1]))

    if std_panel_x_limits[1] <= std_panel_x_limits[0]:
        raise ValueError(
            'predict.counts_bar_plot.std_panel_x_limits must satisfy xmax > xmin.'
        )

    if (
        not isinstance(left_right_width_ratio, (list, tuple))
        or len(left_right_width_ratio) != 2
        or float(left_right_width_ratio[0]) <= 0.0
        or float(left_right_width_ratio[1]) <= 0.0
    ):
        raise ValueError(
            'predict.counts_bar_plot.column_width_ratio must be [left_col_width, right_col_width] with positive values.'
        )

    left_right_width_ratio = [
        float(left_right_width_ratio[0]),
        float(left_right_width_ratio[1]),
    ]

    grid_hspace = float(counts_cfg.get('grid_hspace', 0.35))
    grid_wspace = float(counts_cfg.get('grid_wspace', 0.25))
    left_y_limits_cfg = counts_cfg.get('left_y_limits', None)
    right_y_limits_cfg = counts_cfg.get('right_y_limits', counts_cfg.get('std_y_limits', None))
    align_std_zero_with_left = bool(counts_cfg.get('align_std_zero_with_left', True))

    if left_y_limits_cfg is not None:
        if not isinstance(left_y_limits_cfg, (list, tuple)) or len(left_y_limits_cfg) != 2:
            raise ValueError('predict.counts_bar_plot.left_y_limits must be null or [ymin, ymax].')
        left_y_limits_cfg = (float(left_y_limits_cfg[0]), float(left_y_limits_cfg[1]))

    if right_y_limits_cfg is not None:
        if not isinstance(right_y_limits_cfg, (list, tuple)) or len(right_y_limits_cfg) != 2:
            raise ValueError(
                'predict.counts_bar_plot.right_y_limits (or std_y_limits) must be null or [ymin, ymax].'
            )
        right_y_limits_cfg = (float(right_y_limits_cfg[0]), float(right_y_limits_cfg[1]))

    left_y_label = str(counts_cfg.get('left_y_label', 'Percentage')).strip() or 'Percentage'
    std_y_label = str(counts_cfg.get('std_y_label', 'Percentage STD')).strip() or 'Percentage STD'
    left_x_label = str(counts_cfg.get('left_x_label', 'Physician / Joint')).strip() or 'Physician / Joint'
    std_x_label = str(counts_cfg.get('std_x_label', 'STD across physicians')).strip() or 'STD across physicians'
    panel_title_font_size = float(counts_cfg.get('panel_title_font_size', 10.0))
    title_font_size = float(counts_cfg.get('title_font_size', 12.0))

    title_cfg = counts_cfg.get('title', None)

    if title_cfg is None or not str(title_cfg).strip():
        label = str(target_label).strip()

        if label:
            title = f'{label} predict percentages by subset ({csv_name})'
        else:
            title = f'Predict percentages by subset ({csv_name})'
    else:
        title = str(title_cfg)

    show_plot = bool(counts_cfg.get('show_plot', False))

    def _resolve_panel_pred_labels(source_col):
        if source_col not in df_plot.columns:
            raise ValueError(f'Missing expected prediction column in summary CSV: {source_col}')

        pred_source = pd.to_numeric(df_plot[source_col], errors='coerce')
        pred_binary = pd.Series(np.nan, index=df_plot.index, dtype=float)
        finite_mask = np.isfinite(pred_source)
        pred_binary.loc[finite_mask] = (pred_source.loc[finite_mask] >= 0.5).astype(float)
        return pred_binary

    pred_by_group = {}

    for physician_name in model_physicians:
        pred_by_group[physician_name] = _resolve_panel_pred_labels(physician_name)

    pred_by_group[joint_group_label] = _resolve_panel_pred_labels('JointLabel')

    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(
        n_rows,
        2,
        figure=fig,
        hspace=grid_hspace,
        wspace=grid_wspace,
        width_ratios=left_right_width_ratio,
    )

    left_axes = []
    std_axes = []

    for row_idx, subset_def in enumerate(subset_defs):
        left_ax = fig.add_subplot(gs[row_idx, 0])
        std_ax = fig.add_subplot(gs[row_idx, 1])

        left_axes.append(left_ax)
        std_axes.append(std_ax)

        if subset_def['overall']:
            subset_mask = pd.Series(True, index=df_plot.index)
            match_color = str(counts_cfg.get('overall_match_color', '#4c78a8')).strip() or '#4c78a8'
            mismatch_color = (
                str(counts_cfg.get('overall_mismatch_color', '#9ecae9')).strip() or '#9ecae9'
            )
        else:
            true_label = subset_def['true_label']
            confident = subset_def['confident']
            subset_mask = (df_plot['TrueLabel'] == true_label) & (df_plot['Confident'] == confident)
            match_color = colors[row_idx]
            mismatch_color_idx = subset_index_by_pair.get((1 - true_label, confident), row_idx)
            mismatch_color = colors[mismatch_color_idx]

        subset_label = subset_def['label']

        group_names = list(model_physicians) + [joint_group_label]
        correct_counts = []
        incorrect_counts = []

        for group_name in group_names:
            pred_labels = pred_by_group[group_name]
            valid_pred_mask = pred_labels.isin([0.0, 1.0])
            panel_mask = subset_mask & valid_pred_mask

            if subset_def['overall']:
                true_labels_panel = pd.to_numeric(df_plot.loc[panel_mask, 'TrueLabel'], errors='coerce')
                pred_labels_panel = pred_labels.loc[panel_mask]
                correct_count = int((pred_labels_panel == true_labels_panel).sum())
                incorrect_count = int((pred_labels_panel != true_labels_panel).sum())
            else:
                true_label = subset_def['true_label']
                correct_count = int(((pred_labels == true_label) & panel_mask).sum())
                incorrect_count = int(((pred_labels != true_label) & panel_mask).sum())

            correct_counts.append(correct_count)
            incorrect_counts.append(incorrect_count)

        correct_percentages = []
        incorrect_percentages = []

        for correct_count, incorrect_count in zip(correct_counts, incorrect_counts):
            total_count = correct_count + incorrect_count

            if total_count > 0:
                correct_pct = 100.0 * float(correct_count) / float(total_count)
                incorrect_pct = 100.0 * float(incorrect_count) / float(total_count)
            else:
                correct_pct = 0.0
                incorrect_pct = 0.0

            correct_percentages.append(correct_pct)
            incorrect_percentages.append(incorrect_pct)

        x_positions = np.arange(len(group_names), dtype=float)
        left_ax.bar(
            x_positions,
            correct_percentages,
            width=bar_width,
            color=match_color,
            edgecolor='black',
            linewidth=0.7,
        )
        left_ax.bar(
            x_positions,
            -np.asarray(incorrect_percentages, dtype=float),
            width=bar_width,
            color=mismatch_color,
            edgecolor='black',
            linewidth=0.7,
        )
        left_ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.7)
        left_ax.set_xticks(x_positions)
        left_ax.set_xticklabels(group_names)
        left_ax.set_ylabel(left_y_label)

        if row_idx == (n_rows - 1):
            left_ax.set_xlabel(left_x_label)
        else:
            left_ax.set_xlabel('')

        left_ax.set_title(subset_label, fontsize=panel_title_font_size)
        left_ax.grid(axis='y', alpha=0.25)

        physician_correct = np.asarray(correct_percentages[:-1], dtype=float)
        std_correct = float(np.nanstd(physician_correct, ddof=1)) if physician_correct.size > 1 else 0.0

        std_ax.bar(
            [0.0],
            [std_correct],
            width=std_bar_width,
            color=match_color,
            edgecolor='black',
            linewidth=0.7,
        )
        std_ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.7)
        std_ax.set_xlim(std_panel_x_limits)
        std_ax.set_xticks([0.0])
        std_ax.set_xticklabels(['STD'])
        std_ax.set_ylabel(std_y_label)

        if row_idx == (n_rows - 1):
            std_ax.set_xlabel(std_x_label)
        else:
            std_ax.set_xlabel('')

        std_ax.set_title(f'{subset_label} (STD)', fontsize=panel_title_font_size)
        std_ax.grid(axis='y', alpha=0.25)

    if left_axes:
        if left_y_limits_cfg is None:
            left_min = min(ax.get_ylim()[0] for ax in left_axes)
            left_max = max(ax.get_ylim()[1] for ax in left_axes)
            left_abs = max(abs(left_min), abs(left_max), 1.0)
            left_y_limits = (-left_abs, left_abs)
        else:
            if left_y_limits_cfg[1] <= left_y_limits_cfg[0]:
                raise ValueError('predict.counts_bar_plot.left_y_limits must satisfy ymax > ymin.')
            left_y_limits = left_y_limits_cfg

        for ax in left_axes:
            ax.set_ylim(left_y_limits)

        left_neg = max(-float(left_y_limits[0]), 0.0)
        left_pos = max(float(left_y_limits[1]), 0.0)

        if left_pos <= 0.0:
            left_zero_ratio = 1.0
        else:
            left_zero_ratio = left_neg / left_pos
    else:
        left_zero_ratio = 1.0

    if std_axes:
        std_min_data = min(ax.get_ylim()[0] for ax in std_axes)
        std_max_data = max(ax.get_ylim()[1] for ax in std_axes)
        std_pos_data = max(float(std_max_data), 0.0)
        std_neg_data = max(-float(std_min_data), 0.0)

        if right_y_limits_cfg is None:
            if align_std_zero_with_left and left_zero_ratio > 0.0:
                std_pos_limit = max(std_pos_data, std_neg_data / left_zero_ratio, 1.0)
                std_neg_limit = left_zero_ratio * std_pos_limit
                std_y_limits = (-std_neg_limit, std_pos_limit)
            else:
                std_abs = max(std_neg_data, std_pos_data, 1.0)
                std_y_limits = (-std_abs, std_abs)
        else:
            if right_y_limits_cfg[1] <= right_y_limits_cfg[0]:
                raise ValueError(
                    'predict.counts_bar_plot.right_y_limits (or std_y_limits) must satisfy ymax > ymin.'
                )

            if align_std_zero_with_left and left_zero_ratio > 0.0:
                cfg_neg = max(-float(right_y_limits_cfg[0]), 0.0)
                cfg_pos = max(float(right_y_limits_cfg[1]), 0.0)
                std_pos_limit = max(cfg_pos, std_pos_data, std_neg_data / left_zero_ratio)
                std_neg_limit = max(left_zero_ratio * std_pos_limit, cfg_neg)
                std_y_limits = (-std_neg_limit, std_pos_limit)
            else:
                std_y_limits = right_y_limits_cfg

        for ax in std_axes:
            ax.set_ylim(std_y_limits)

    fig.suptitle(str(title), y=float(counts_cfg.get('title_y', 0.995)), fontsize=title_font_size)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close(fig)


def _draw_physician_ridge_grid(
    df_plot,
    model_physicians,
    out_png,
    csv_name,
    target_label,
    model_ignore_confidence,
    ridge_cfg,
    joint_method
):
    """Render and save a physician subplot grid of stacked ridge plots."""
    style = str(ridge_cfg.get('style', 'white')).strip() or 'white'
    context = str(ridge_cfg.get('context', 'paper')).strip() or 'paper'
    sns.set_theme(style=style, context=context)

    joint_confidence_panel_label = str(
        ridge_cfg.get('joint_confidence_panel_label', 'JointConfidence')
    ).strip() or 'JointConfidence'

    method_token = 'LR' if joint_method == 'joint_LR' else \
            ('Bayes' if joint_method == 'bayes' else '??')

    joint_confidence_panel_label = f'{joint_confidence_panel_label} ({method_token})' 

    std_confidence_panel_label = str(
        ridge_cfg.get('std_confidence_panel_label', 'STDConfidence over model physicians')
    ).strip() or 'STDConfidence over model physicians'

    panel_defs = [
        {
            'label': physician_name,
            'source_col': f'{physician_name}_PredConfidence',
            'pred_source_col': physician_name,
            'is_aggregate': False,
        }
        for physician_name in model_physicians
    ]
    panel_defs.extend([
        {
            'label': joint_confidence_panel_label,
            'source_col': 'JointConfidence',
            'pred_source_col': 'JointLabel',
            'is_aggregate': True,
        },
        {
            'label': std_confidence_panel_label,
            'source_col': 'STDConfidence',
            'pred_source_col': 'JointLabel',
            'is_aggregate': True,
            'fallback_cols': [f'{physician_name}_PredConfidence' for physician_name in model_physicians],
            'fallback_agg': 'std',
            'fallback_pred_cols': model_physicians,
        },
    ])

    n_panels = len(panel_defs)
    n_cols = int(ridge_cfg.get('n_cols', min(3, max(1, n_panels))))
    n_cols = max(1, min(n_cols, n_panels))
    n_rows = int(math.ceil(n_panels / n_cols))

    figure_size = ridge_cfg.get('figure_size', None)

    if figure_size is None:
        width_per_cell = float(ridge_cfg.get('width_per_physician', 4.6))
        height_per_cell = float(ridge_cfg.get('height_per_physician', 5.2))
        figure_size = (n_cols * width_per_cell, n_rows * height_per_cell)
    else:
        if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
            raise ValueError('predict.ridge_plots.figure_size must be null or [width, height].')

        figure_size = (float(figure_size[0]), float(figure_size[1]))

    dpi = int(ridge_cfg.get('dpi', 300))
    x_limits = ridge_cfg.get('x_limits', [0.0, 1.0])

    if x_limits is None:
        x_limits = (0.0, 1.0)
    else:
        if not isinstance(x_limits, (list, tuple)) or len(x_limits) != 2:
            raise ValueError('predict.ridge_plots.x_limits must be null or [xmin, xmax].')

        x_limits = (float(x_limits[0]), float(x_limits[1]))

    std_x_limits = ridge_cfg.get('std_x_limits', [0.0, 1.0])

    if std_x_limits is None:
        std_x_limits = (0.0, 1.0)
    else:
        if not isinstance(std_x_limits, (list, tuple)) or len(std_x_limits) != 2:
            raise ValueError('predict.ridge_plots.std_x_limits must be null or [xmin, xmax].')

        std_x_limits = (float(std_x_limits[0]), float(std_x_limits[1]))

    colors = ridge_cfg.get('colors', ['#1a9850', '#91cf60', '#fc8d59', '#d73027'])

    if not isinstance(colors, (list, tuple)) or len(colors) < 4:
        raise ValueError('predict.ridge_plots.colors must contain at least 4 colors.')

    subset_labels_cfg = ridge_cfg.get('subset_labels', {}) or {}
    subset_defs = [
        {
            'true_label': 0,
            'confident': 1,
            'label': subset_labels_cfg.get('0_1', 'True=0, Confident=1'),
            'overall': False,
        },
        {
            'true_label': 0,
            'confident': 0,
            'label': subset_labels_cfg.get('0_0', 'True=0, Confident=0'),
            'overall': False,
        },
        {
            'true_label': 1,
            'confident': 0,
            'label': subset_labels_cfg.get('1_0', 'True=1, Confident=0'),
            'overall': False,
        },
        {
            'true_label': 1,
            'confident': 1,
            'label': subset_labels_cfg.get('1_1', 'True=1, Confident=1'),
            'overall': False,
        },
        {
            'true_label': None,
            'confident': None,
            'label': subset_labels_cfg.get('overall', 'Overall'),
            'overall': True,
        },
    ]
    n_subset_rows = len(subset_defs)

    kde_bw_adjust = float(ridge_cfg.get('kde_bw_adjust', 0.9))
    fill_alpha = float(ridge_cfg.get('fill_alpha', 0.8))
    line_width = float(ridge_cfg.get('line_width', 1.2))
    mismatch_density_scale_cfg = ridge_cfg.get('mismatch_density_scale', 'auto')
    symmetric_signed_y = bool(ridge_cfg.get('symmetric_signed_y', True))
    kde_n_points = max(64, int(ridge_cfg.get('kde_n_points', 256)))
    single_value_height = float(ridge_cfg.get('single_value_height', 1.0))

    mismatch_density_scale_auto = False
    mismatch_density_scale = 1.0

    if isinstance(mismatch_density_scale_cfg, str):
        if mismatch_density_scale_cfg.strip().lower() == 'auto':
            mismatch_density_scale_auto = True
        else:
            try:
                mismatch_density_scale = float(mismatch_density_scale_cfg)
            except ValueError as err:
                raise ValueError(
                    'predict.ridge_plots.mismatch_density_scale must be "auto" or a non-negative number.'
                ) from err
    else:
        mismatch_density_scale = float(mismatch_density_scale_cfg)

    if mismatch_density_scale < 0.0:
        raise ValueError('predict.ridge_plots.mismatch_density_scale must be >= 0.')

    outer_hspace = float(ridge_cfg.get('grid_hspace', 0.28))
    outer_wspace = float(ridge_cfg.get('grid_wspace', 0.18))
    inner_hspace = float(ridge_cfg.get('ridge_hspace', -0.35))

    title_cfg = ridge_cfg.get('title', None)
    model_title_suffix = (
        ' - non-confident (soft) models'
        if model_ignore_confidence
        else ' - confident (strict) models'
    )

    if title_cfg is None or not str(title_cfg).strip():
        title_label = str(target_label).strip()

        if title_label:
            title = f'{title_label}{model_title_suffix}'
        else:
            title = f'Prediction confidence ridge plots by physician{model_title_suffix}\n{csv_name}'
    else:
        title = str(title_cfg)

    show_plot = bool(ridge_cfg.get('show_plot', False))
    show_x_axis_on_all_ridges = bool(ridge_cfg.get('show_x_axis_on_all_ridges', False))
    show_y_axis_on_leftmost_ridges = bool(
        ridge_cfg.get('show_y_axis_on_leftmost_ridges', False)
    )
    show_pane_y_label = bool(ridge_cfg.get('show_pane_y_label', True))
    pane_y_label = str(ridge_cfg.get('pane_y_label', 'Signed density')).strip() or 'Signed density'
    ridge_hspace_with_all_x_axes = float(ridge_cfg.get('ridge_hspace_with_all_x_axes', 0.08))
    pane_x_label = str(ridge_cfg.get('pane_x_label', 'PredConfidence'))
    std_x_label = str(ridge_cfg.get('std_x_label', pane_x_label))
    y_limits_cfg = ridge_cfg.get('y_limits', None)

    inner_hspace_effective = inner_hspace

    if show_x_axis_on_all_ridges:
        # Overlapping ridges can hide x-axes; enforce non-overlapping spacing in this mode.
        inner_hspace_effective = max(inner_hspace, ridge_hspace_with_all_x_axes)

    if y_limits_cfg is not None:
        if not isinstance(y_limits_cfg, (list, tuple)) or len(y_limits_cfg) != 2:
            raise ValueError('predict.ridge_plots.y_limits must be null or [ymin, ymax].')

        y_limits_cfg = (
            None if y_limits_cfg[0] is None else float(y_limits_cfg[0]),
            None if y_limits_cfg[1] is None else float(y_limits_cfg[1]),
        )

    fig = plt.figure(figsize=figure_size)
    outer = GridSpec(n_rows, n_cols, figure=fig, hspace=outer_hspace, wspace=outer_wspace)
    ridge_axes = []

    physician_conf_cols = [f'{physician_name}_PredConfidence' for physician_name in model_physicians]

    for col_name in physician_conf_cols:
        if col_name not in df_plot.columns:
            raise ValueError(
                f'Missing expected confidence column in summary CSV: {col_name}'
            )

    def _resolve_panel_values(source_col, fallback_cols=None, fallback_agg=None):
        if source_col in df_plot.columns:
            return pd.to_numeric(df_plot[source_col], errors='coerce')

        if not fallback_cols:
            raise ValueError(
                f'Missing expected confidence column in summary CSV: {source_col}'
            )

        fallback_frame = df_plot[fallback_cols].apply(pd.to_numeric, errors='coerce')

        if fallback_agg == 'mean':
            return fallback_frame.mean(axis=1, skipna=True)

        if fallback_agg == 'std':
            return fallback_frame.std(axis=1, skipna=True, ddof=1)

        raise ValueError(f'Unsupported fallback aggregate: {fallback_agg!r}')

    def _resolve_panel_pred_labels(panel_def):
        source_col = panel_def.get('pred_source_col')

        if source_col in df_plot.columns:
            pred_source = pd.to_numeric(df_plot[source_col], errors='coerce')
        else:
            fallback_pred_cols = panel_def.get('fallback_pred_cols')

            if not fallback_pred_cols:
                raise ValueError(
                    f'Missing expected prediction column in summary CSV: {source_col}'
                )

            pred_source = df_plot[fallback_pred_cols].apply(
                pd.to_numeric,
                errors='coerce',
            ).mean(axis=1, skipna=True)

        pred_binary = pd.Series(np.nan, index=df_plot.index, dtype=float)
        finite_mask = np.isfinite(pred_source)
        pred_binary.loc[finite_mask] = (pred_source.loc[finite_mask] >= 0.5).astype(float)
        return pred_binary

    def _density_curve(vals, panel_x_limits):
        if vals.size < 2 or np.unique(vals).size < 2:
            return None, None

        x_grid = np.linspace(panel_x_limits[0], panel_x_limits[1], kde_n_points)

        try:
            kde = gaussian_kde(vals)
            bw_factor = max(kde.factor * max(kde_bw_adjust, 1e-3), 1e-9)
            kde.set_bandwidth(bw_method=bw_factor)
            density = kde.evaluate(x_grid)
        except Exception:
            n_bins = max(10, min(60, int(np.sqrt(vals.size))))
            hist, bins = np.histogram(
                vals,
                bins=n_bins,
                range=panel_x_limits,
                density=True,
            )
            centers = 0.5 * (bins[:-1] + bins[1:])
            density = np.interp(x_grid, centers, hist, left=0.0, right=0.0)

        density = np.clip(density, 0.0, None)
        return x_grid, density

    def _draw_signed_distribution(
        ax,
        vals,
        panel_x_limits,
        color,
        sign,
        density_scale,
    ):
        if vals.size == 0:
            return

        x_grid, density = _density_curve(vals, panel_x_limits)
        scale = max(float(density_scale), 0.0)

        if x_grid is not None:
            signed_density = density * scale if sign > 0 else -(density * scale)
            ax.fill_between(
                x_grid,
                0.0,
                signed_density,
                facecolor=color,
                edgecolor=color,
                alpha=fill_alpha,
                linewidth=line_width,
            )

            ax.plot(x_grid, signed_density, color=color, linewidth=line_width)
            return

        if vals.size == 1 and np.isfinite(vals[0]):
            tip_height = single_value_height * scale
            tip = tip_height if sign > 0 else -tip_height
            ax.vlines(
                vals[0],
                0.0,
                tip,
                color=color,
                linewidth=line_width + 0.4,
                linestyles='-' if sign > 0 else '--',
            )

    def _panel_limits(panel_def):
        if panel_def['source_col'] == 'STDConfidence':
            return std_x_limits

        return x_limits

    subset_index_by_pair = {
        (subset_def['true_label'], subset_def['confident']): idx
        for idx, subset_def in enumerate(subset_defs)
        if not subset_def['overall']
    }

    for idx, panel_def in enumerate(panel_defs):
        row = idx // n_cols
        col = idx % n_cols

        inner = GridSpecFromSubplotSpec(
            n_subset_rows,
            1,
            subplot_spec=outer[row, col],
            hspace=inner_hspace_effective,
        )

        vals_all = _resolve_panel_values(
            panel_def['source_col'],
            fallback_cols=panel_def.get('fallback_cols'),
            fallback_agg=panel_def.get('fallback_agg'),
        )
        pred_labels_all = _resolve_panel_pred_labels(panel_def)
        panel_x_limits = _panel_limits(panel_def)

        for i_subset, subset_def in enumerate(subset_defs):
            ax = fig.add_subplot(inner[i_subset, 0])
            ridge_axes.append(ax)

            if not show_x_axis_on_all_ridges:
                # Keep overlaps readable: remove opaque subplot backgrounds.
                ax.set_facecolor((0, 0, 0, 0))

            valid_pred_mask = pred_labels_all.isin([0.0, 1.0])

            if subset_def['overall']:
                subset_mask = pd.Series(True, index=df_plot.index)
                true_labels_all = pd.to_numeric(df_plot['TrueLabel'], errors='coerce')
                match_mask = subset_mask & valid_pred_mask & (pred_labels_all == true_labels_all)
                mismatch_mask = subset_mask & valid_pred_mask & (pred_labels_all != true_labels_all)
                match_color = str(ridge_cfg.get('overall_match_color', '#4c78a8')).strip() or '#4c78a8'
                mismatch_color = (
                    str(ridge_cfg.get('overall_mismatch_color', '#9ecae9')).strip() or '#9ecae9'
                )
            else:
                true_label = subset_def['true_label']
                confident = subset_def['confident']
                subset_mask = (df_plot['TrueLabel'] == true_label) & (df_plot['Confident'] == confident)
                match_mask = subset_mask & valid_pred_mask & (pred_labels_all == true_label)
                mismatch_mask = subset_mask & valid_pred_mask & (pred_labels_all != true_label)
                match_color = colors[i_subset]
                mismatch_color_index = subset_index_by_pair.get((1 - true_label, confident), i_subset)
                mismatch_color = colors[mismatch_color_index]

            subset_label = subset_def['label']

            vals_match = vals_all.loc[match_mask].dropna().to_numpy(dtype=float)
            vals_match = vals_match[np.isfinite(vals_match)]

            vals_mismatch = vals_all.loc[mismatch_mask].dropna().to_numpy(dtype=float)
            vals_mismatch = vals_mismatch[np.isfinite(vals_mismatch)]

            if mismatch_density_scale_auto:
                n_total = vals_match.size + vals_mismatch.size

                if n_total > 0:
                    scale_match = vals_match.size / n_total
                    scale_mismatch = vals_mismatch.size / n_total
                else:
                    scale_match = 0.0
                    scale_mismatch = 0.0
            else:
                scale_match = 1.0
                scale_mismatch = mismatch_density_scale

            _draw_signed_distribution(
                ax=ax,
                vals=vals_match,
                panel_x_limits=panel_x_limits,
                color=match_color,
                sign=1,
                density_scale=scale_match,
            )
            _draw_signed_distribution(
                ax=ax,
                vals=vals_mismatch,
                panel_x_limits=panel_x_limits,
                color=mismatch_color,
                sign=-1,
                density_scale=scale_mismatch,
            )

            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.65)
            ax.set_xlim(panel_x_limits)

            if show_pane_y_label and (col == 0):
                ax.set_ylabel(pane_y_label)
            else:
                ax.set_ylabel('')

            show_this_y_axis = show_y_axis_on_leftmost_ridges and (col == 0)

            if show_this_y_axis:
                ax.spines['left'].set_visible(True)
                ax.tick_params(axis='y', which='both', left=True, labelleft=True)
            else:
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

            show_this_x_axis = show_x_axis_on_all_ridges or (i_subset == (n_subset_rows - 1))

            if show_this_x_axis:
                ax.spines['bottom'].set_visible(True)
                if i_subset == (n_subset_rows - 1):
                    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
                else:
                    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    ax.set_xticklabels([])
            else:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                ax.set_xticklabels([])

            if i_subset == (n_subset_rows - 1):
                panel_x_label = std_x_label if panel_def['source_col'] == 'STDConfidence' else pane_x_label
                ax.set_xlabel(panel_x_label)
            else:
                ax.set_xlabel('')

            label_x = 0.01
            label_ha = 'left'
            subset_font_size = float(ridge_cfg.get('subset_font_size', 8.0))

            ax.text(
                label_x,
                0.95,
                subset_label,
                transform=ax.transAxes,
                fontsize=subset_font_size,
                ha=label_ha,
                va='top',
            )

            ax.text(
                label_x,
                0.82,
                f'+n={vals_match.size}, -n={vals_mismatch.size}',
                transform=ax.transAxes,
                fontsize=subset_font_size,
                ha=label_ha,
                va='top',
            )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if i_subset == 0:
                title_prefix = (
                    str(ridge_cfg.get('mean_title_prefix', 'Aggregate: '))
                    if panel_def['is_aggregate']
                    else str(ridge_cfg.get('physician_title_prefix', 'Model physician: '))
                )
                ax.set_title(
                    title_prefix + panel_def['label'],
                    fontsize=float(ridge_cfg.get('physician_title_font_size', 10.0)),
                    pad=2.0,
                )

    if ridge_axes:
        auto_ymin = min(ax.get_ylim()[0] for ax in ridge_axes)
        auto_ymax = max(ax.get_ylim()[1] for ax in ridge_axes)

        if not np.isfinite(auto_ymin) or not np.isfinite(auto_ymax) or auto_ymax <= auto_ymin:
            auto_ymin, auto_ymax = 0.0, 1.0

        if y_limits_cfg is None and symmetric_signed_y:
            max_abs = max(abs(auto_ymin), abs(auto_ymax))

            if not np.isfinite(max_abs) or max_abs <= 0.0:
                max_abs = 1.0

            auto_ymin, auto_ymax = -max_abs, max_abs

        if y_limits_cfg is None:
            shared_ymin, shared_ymax = auto_ymin, auto_ymax
        else:
            shared_ymin = auto_ymin if y_limits_cfg[0] is None else y_limits_cfg[0]
            shared_ymax = auto_ymax if y_limits_cfg[1] is None else y_limits_cfg[1]

            if shared_ymax <= shared_ymin:
                raise ValueError('predict.ridge_plots.y_limits must satisfy ymax > ymin.')

        for ax in ridge_axes:
            ax.set_ylim(shared_ymin, shared_ymax)

    # Empty slots in the outer physician grid should stay blank.
    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(outer[row, col])
        ax.axis('off')

    fig.suptitle(
        str(title),
        y=float(ridge_cfg.get('title_y', 0.995)),
        fontsize=float(ridge_cfg.get('title_font_size', 12.0)),
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close(fig)
