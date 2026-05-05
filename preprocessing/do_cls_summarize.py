"""
**Summarize classification results from saved data.**
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
from pandas.errors import EmptyDataError

STEP = 'summarize'


def do_cls_summarize(ss):
    """
    Collect data from the .pkl files and summarize main results
    in a single DataFrame/.csv file.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    out_root = Path(ss.out_root)
    summary_columns = ss.args.get(
        'summary_columns',
        [
            'Hospital','Physician','Label','dim_reduction','nparms','ignore_confidence','standardize',
            'CV folds', 'seed',
            'Accuracy', 'Precision N', 'Recall N', 'f1-N',
            'Precision P', 'Recall P', 'f1-P', 'AUC'
        ]
    )

    if 'dim_reduction' not in summary_columns:
        if 'Label' in summary_columns:
            label_pos = summary_columns.index('Label')
            summary_columns = summary_columns[:label_pos + 1] + ['dim_reduction'] + summary_columns[label_pos + 1:]
        else:
            summary_columns = ['dim_reduction'] + summary_columns

    # Collect all saved classification pickles in out_root, regardless of hospital token (hlist)
    pattern = 'xgb_*_nparms*_std*_ignoreConf*.pkl'
    pkl_files = [*out_root.glob(pattern)]

    if not pkl_files:
        print(f'No classification pickle files found in {out_root} using pattern: {pattern}')
        print(f'\n Step *{STEP}* completed successfully')
        return

    rows = []
    for pkl_pname in pkl_files:
        try:
            with open(pkl_pname, 'rb') as fp:
                payload = pickle.load(fp)
        except Exception as exc:
            print(f'Warning: failed to read {pkl_pname}: {exc}')
            continue

        Y = payload.get('Y')
        y_pred = payload.get('y_pred')
        y_score = payload.get('y_score')

        if Y is None or y_pred is None:
            print(f'Warning: skipping {pkl_pname.name} (missing Y or y_pred)')
            continue

        Y = np.asarray(Y)
        y_pred = np.asarray(y_pred)

        # metadata
        hospitals = payload.get('hospital', ss.args.get('hospital', []))
        hospital_token = ss.hlist(hospitals) if isinstance(hospitals, list) and hospitals else ''
        physician = payload.get('physician')
        physician_cell = _physician_to_cell(physician)
        target_label = payload.get('target_label', ss.args.get('target_label'))
        ignore_confidence = payload.get('ignore_confidence', ss.args.get('ignore_confidence'))
        standardize = payload.get('standardize', ss.args.get('standardize_features', False))
        seed = payload.get('seed', ss.args.get('seed'))
        cv_folds = payload.get('cv_n_splits', ss.args.get('cv_n_splits'))

        # infer nparms from full model reducer feature shape if available; otherwise filename
        nparms = None
        fmd = payload.get('full_model_data') if isinstance(payload, dict) else None

        if isinstance(fmd, dict):
            reducer = fmd.get('reducer')
            method = reducer.get('method') if isinstance(reducer, dict) else None

            if method in ('none', 'pca', 'to_lobes'):
                # nparms is still stored in filename; keep this branch as future-proof fallback
                pass

        if nparms is None:          # Infer from the file name
            stem = pkl_pname.stem
            token = '_nparms'

            if token in stem:
                tail = stem.split(token, 1)[1]
                nparms_txt = tail.split('_', 1)[0]
                try:
                    nparms = int(nparms_txt)
                except ValueError:
                    nparms = np.nan
            else:
                nparms = np.nan

        dim_reduction = None

        # Now try to deduce the dim reduction method used for the
        # pickle
        if isinstance(fmd, dict):
            dim_reduction = fmd.get('dim_reduction')    # Hopefully, it is just stored there

            if dim_reduction is None:
                # If not stored, get it from the stored reducer data
                reducer = fmd.get('reducer')

                if isinstance(reducer, dict):
                    dim_reduction = reducer.get('method')

        # If everything fails, just use the one currently specified in JSON
        if dim_reduction is None:
            dim_reduction = payload.get('dim_reduction', ss.args.get('dim_reduction', 'unknown'))
            print(
                'Warning: dim reduction method could not be found in '
                f'{pkl_pname.name}; using current JSON/default setting in summary output: {dim_reduction}'
            )

        # classification metrics
        acc = accuracy_score(Y, y_pred)

        # This is what CV results report usually contains
        pr, rc, f1, _ = precision_recall_fscore_support(
            Y,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )

        # Get AUC of PR curve
        auc_val = np.nan

        if y_score is not None:
            y_score = np.asarray(y_score)

            if np.unique(Y).size == 2 and y_score.shape[0] == Y.shape[0]:
                try:
                    auc_val = average_precision_score(Y, y_score)
                except Exception:
                    auc_val = np.nan

        row = {
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
            'AUC': auc_val,
        }

        rows.append(row)        # Append to the list of rows; there is no df yet

    if not rows:
        print(f'No valid classification entries could be extracted from {len(pkl_files)} file(s).')
        print(f'\n Step *{STEP}* completed successfully')
        return

    df = pd.DataFrame(rows)     # Create df with hard coded column names for now

    # Sort rows as requested:
    # 1) Label alphabetically
    # 2) ignore_confidence: True first, then False
    # 3) nparms: 5 first, then 9

    # Sort dataframe rows in order of: label; ignore_confidence; nparms
    df['_label_sort'] = df['Label'].astype(str).str.lower()
    df['_ignore_sort'] = df['ignore_confidence'].map(lambda v: 0 if _to_bool(v) else 1)
    df['_nparms_sort'] = df['nparms'].map(_nparms_sort_key)

    df = df.sort_values(
        by=['_label_sort', '_ignore_sort', '_nparms_sort'],
        ascending=[True, True, True],
        kind='mergesort',
    ).reset_index(drop=True)

    df = df.drop(columns=['_label_sort', '_ignore_sort', '_nparms_sort'])

    # Rename columns to those given in the JSON
    df = df.reindex(columns=summary_columns)

    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    summary_csv_name = ss.args['summary_csv_name']
    csv_pname = out_root / summary_csv_name
    csv_pname.parent.mkdir(parents=True, exist_ok=True)

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
            # Align by column name to avoid value/header shifts when appending
            # to an existing summary produced with a different schema.
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
            print(
                'Existing summary CSV columns differ from current summary schema; '
                'rewrote file with aligned columns to prevent shifted values.'
            )
    else:
        df.to_csv(
            csv_pname,
            mode='w',
            header=True,
            index=False,
            float_format='%.3f',
            quoting=csv.QUOTE_ALL,
        )

    if wrote_mode == 'appended':
        print(f'Summary appended to: {csv_pname}')
    elif wrote_mode == 'rewritten':
        print(f'Summary rewritten with aligned columns: {csv_pname}')
    else:
        print(f'Summary saved to: {csv_pname}')

    print(f'\n Step *{STEP}* completed successfully')


def _physician_to_cell(value):
    if value is None:
        return ''

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple, set)):
        names = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ', '.join(names)

    return str(value).strip()


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 't', 'yes', 'y')
    return bool(v)


def _nparms_sort_key(v):
    try:
        nv = int(v)
    except Exception:
        return (2, 9999)
    if nv == 5:
        return (0, nv)
    if nv == 9:
        return (1, nv)
    return (2, nv)
