"""
Shared helpers for classification summary rows and summary CSV I/O.
"""
import csv
from pathlib import Path

import fcntl        # Built-in package for using advisory file blocks
from contextlib import contextmanager   # Decorator which adds a custom context to a
                                        # generator function - to be used with 'with' block

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
)

dummy_BS = lambda p: p * (1 - p)  # noqa: E731  # mirrors cls_calibrate.dummy_BS


DEFAULT_SUMMARY_COLUMNS = [
    'Hospital',
    'Physician',
    'Label',
    'dim_reduction',
    'nparms',
    'ignore_confidence',
    'standardize',
    'CV folds',
    'seed',
    'Accuracy',
    'Precision N',
    'Recall N',
    'f1-N',
    'Precision P',
    'Recall P',
    'f1-P',
    'Calibration',
    'AUC',
    'BS',
    'Dummy BS',
]


def get_summary_columns(ss):
    """Get summary CSV columns with backward-compatible dim_reduction insertion."""
    summary_columns = ss.args.get('summary_columns', DEFAULT_SUMMARY_COLUMNS)

    if 'dim_reduction' not in summary_columns:
        if 'Label' in summary_columns:
            label_pos = summary_columns.index('Label')
            summary_columns = summary_columns[:label_pos + 1] + ['dim_reduction'] + summary_columns[label_pos + 1:]
        else:
            summary_columns = ['dim_reduction'] + summary_columns

    if 'AUC' in summary_columns:
        auc_pos = summary_columns.index('AUC')

        if 'Calibration' not in summary_columns:
            summary_columns = summary_columns[:auc_pos] + ['Calibration'] + summary_columns[auc_pos:]
            auc_pos += 1

        if 'BS' not in summary_columns:
            summary_columns = summary_columns[:auc_pos + 1] + ['BS'] + summary_columns[auc_pos + 1:]

        bs_pos = summary_columns.index('BS')

        if 'Dummy BS' not in summary_columns:
            summary_columns = summary_columns[:bs_pos + 1] + ['Dummy BS'] + summary_columns[bs_pos + 1:]
    else:
        if 'Calibration' not in summary_columns:
            summary_columns = summary_columns + ['Calibration']

        if 'BS' not in summary_columns:
            summary_columns = summary_columns + ['BS']

        if 'Dummy BS' not in summary_columns:
            summary_columns = summary_columns + ['Dummy BS']

    return summary_columns


def extract_nparms_from_pickle_name(pkl_pname):
    """Infer nparms from pickle file name token _nparms<N>."""
    stem = Path(pkl_pname).stem
    token = '_nparms'

    if token in stem:
        tail = stem.split(token, 1)[1]
        nparms_txt = tail.split('_', 1)[0]
        try:
            return int(nparms_txt)
        except ValueError:
            return np.nan

    return np.nan


def infer_dim_reduction(payload, pkl_pname, ss):
    """Infer dim reduction from pickle payload, with JSON/default fallback."""
    dim_reduction = None
    fmd = payload.get('full_model_data') if isinstance(payload, dict) else None

    if isinstance(fmd, dict):
        dim_reduction = fmd.get('dim_reduction')

        if dim_reduction is None:
            reducer = fmd.get('reducer')

            if isinstance(reducer, dict):
                dim_reduction = reducer.get('method')

    if dim_reduction is None:
        dim_reduction = payload.get('dim_reduction', ss.args.get('dim_reduction', 'unknown'))
        print(
            'Warning: dim reduction method could not be found in '
            f'{Path(pkl_pname).name}; using current JSON/default setting in summary output: {dim_reduction}'
        )

    return dim_reduction


def build_summary_row(
    ss,
    payload,
    pkl_pname,
    y_true,
    y_pred,
    y_score=None,
    physician_cell=None,
    calibration='none',
):
    """Build one summary row dictionary from labels/scores and pickle metadata."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    hospitals = payload.get('hospital', ss.args.get('hospital', []))
    hospital_token = ss.hlist(hospitals) if isinstance(hospitals, list) and hospitals else ''
    target_label = payload.get('target_label', ss.args.get('target_label'))
    ignore_confidence = payload.get('ignore_confidence', ss.args.get('ignore_confidence'))
    standardize = payload.get('standardize', ss.args.get('standardize_features', False))
    seed = payload.get('seed', ss.args.get('seed'))
    cv_folds = payload.get('cv_n_splits', ss.args.get('cv_n_splits'))

    if physician_cell is None:
        physician_cell = physician_to_cell(payload.get('physician'))

    nparms = extract_nparms_from_pickle_name(pkl_pname)
    dim_reduction = infer_dim_reduction(payload, pkl_pname, ss)

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    auc_val = np.nan
    bs_val = np.nan
    dummy_bs_val = np.nan

    if np.unique(y_true).size == 2 and y_true.shape[0] > 0:
        pos_ratio = float(np.sum(y_true == 1)) / float(y_true.shape[0])
        try:
            dummy_bs_val = dummy_BS(pos_ratio)
        except Exception:
            dummy_bs_val = np.nan

    if y_score is not None:
        y_score = np.asarray(y_score)

        if np.unique(y_true).size == 2 and y_score.shape[0] == y_true.shape[0]:
            try:
                auc_val = average_precision_score(y_true, y_score)
            except Exception:
                auc_val = np.nan

            try:
                bs_val = brier_score_loss(y_true, y_score)
            except Exception:
                bs_val = np.nan

    return {
        'Hospital': hospital_token,
        'Physician': physician_cell,
        'Label': target_label,
        'dim_reduction': dim_reduction,
        'nparms': nparms,
        'ignore_confidence': ignore_confidence,
        'standardize': standardize,
        'CV folds': cv_folds,
        'seed': seed,
        'Accuracy': acc,
        'Precision N': pr[0],
        'Recall N': rc[0],
        'f1-N': f1[0],
        'Precision P': pr[1],
        'Recall P': rc[1],
        'f1-P': f1[1],
        'Calibration': calibration,
        'AUC': auc_val,
        'BS': bs_val,
        'Dummy BS': dummy_bs_val,
    }


def align_summary_dataframe(df, summary_columns):
    """Align DataFrame to configured summary columns."""
    return df.reindex(columns=summary_columns)


def _summary_csv_lock_path(csv_pname):
    """
    Return 'sidecar' lock file path for a summary CSV. This simply
    will be <CSV-name>.lock. In fact, no lock file will be created - rather,
    corresponding entry will be added directly to the inode of the CSV file in
    the system-wide Open File table.
    """
    csv_pname = Path(csv_pname)
    return csv_pname.with_name(f'{csv_pname.name}.lock')

# Explanation for the code below:
# @contextmanager decorator is a shortcut to create a function
# using its own context, so this function could be called using
# the 'with' statement.
@contextmanager
def _summary_csv_lock(csv_pname):
    """
    Acquire an exclusive cross-process advisory lock for summary CSV writes.
    Note that _summary_csv_lock() function is not needed, because unlock
    happens automatically when the 'with' block is exited.
    """
    lock_path = _summary_csv_lock_path(csv_pname)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, 'a+', encoding='utf-8') as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)    # Aquire exclusive lock

        try:
            yield       # Return control to the caller to access the intended file
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)    # Unlock in case of error

def append_or_merge_summary_csv(csv_pname, df, summary_columns):
    """
    Save summary data to CSV with schema-aware append/merge behavior.

    Returns one of: saved, appended, rewritten.
    """
    csv_pname = Path(csv_pname)
    csv_pname.parent.mkdir(parents=True, exist_ok=True)

    with _summary_csv_lock(csv_pname):
        append_existing = csv_pname.exists() and csv_pname.stat().st_size > 0
        wrote_mode = 'saved'

        if append_existing:
            try:
                existing_df = pd.read_csv(csv_pname)
            except EmptyDataError:
                existing_df = pd.DataFrame(columns=df.columns)

            if list(existing_df.columns) == list(df.columns):
                df.to_csv(
                    csv_pname,
                    mode='a',
                    header=False,
                    index=False,
                    float_format='%.3f',
                    quoting=csv.QUOTE_ALL,
                )
                wrote_mode = 'appended'
            else:
                existing_aligned = existing_df.reindex(columns=summary_columns)
                new_aligned = df.reindex(columns=summary_columns)
                combined_df = pd.concat([existing_aligned, new_aligned], ignore_index=True)
                combined_df.to_csv(
                    csv_pname,
                    mode='w',
                    header=True,
                    index=False,
                    float_format='%.3f',
                    quoting=csv.QUOTE_ALL,
                )
                wrote_mode = 'rewritten'
        else:
            df.to_csv(
                csv_pname,
                mode='w',
                header=True,
                index=False,
                float_format='%.3f',
                quoting=csv.QUOTE_ALL,
            )

    return wrote_mode


def physician_to_cell(value):
    """Convert physician field value to summarize-step cell string."""
    if value is None:
        return ''

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple, set)):
        names = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ', '.join(names)

    return str(value).strip()


def physician_token(value):
    """Convert physician value into token form for filenames and model2target values."""
    if value is None:
        return 'All'

    if isinstance(value, str):
        token = value.strip()
        return token if token else 'All'

    if isinstance(value, (list, tuple, set)):
        names = [str(v).strip() for v in value if v is not None and str(v).strip()]

        if not names:
            return 'All'

        return '_'.join(names)

    token = str(value).strip()
    return token if token else 'All'


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 't', 'yes', 'y')
    return bool(v)


def nparms_sort_key(v):
    try:
        nv = int(v)
    except Exception:
        return (2, 9999)
    if nv == 5:
        return (0, nv)
    if nv == 9:
        return (1, nv)
    return (2, nv)
