"""
**Summarize classification results from saved data.**
"""
import pandas as pd
from pathlib import Path
import pickle

from cls_summary_utils import (
    align_summary_dataframe,
    append_or_merge_summary_csv,
    build_summary_row,
    get_summary_columns,
    nparms_sort_key,
    physician_to_cell,
    to_bool,
)

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
    summary_columns = get_summary_columns(ss)

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

        y_true = payload.get('Y')
        y_pred = payload.get('y_pred')
        y_score = payload.get('y_score')

        if y_true is None or y_pred is None:
            print(f'Warning: skipping {pkl_pname.name} (missing Y or y_pred)')
            continue

        rows.append(
            build_summary_row(
                ss=ss,
                payload=payload,
                pkl_pname=pkl_pname,
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                physician_cell=physician_to_cell(payload.get('physician')),
            )
        )

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
    df['_ignore_sort'] = df['ignore_confidence'].map(lambda v: 0 if to_bool(v) else 1)
    df['_nparms_sort'] = df['nparms'].map(nparms_sort_key)

    df = df.sort_values(
        by=['_label_sort', '_ignore_sort', '_nparms_sort'],
        ascending=[True, True, True],
        kind='mergesort',
    ).reset_index(drop=True)

    df = df.drop(columns=['_label_sort', '_ignore_sort', '_nparms_sort'])

    # Rename columns to those given in the JSON
    df = align_summary_dataframe(df, summary_columns)

    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    summary_csv_name = ss.args['summary_csv_name']
    csv_pname = out_root / summary_csv_name
    wrote_mode = append_or_merge_summary_csv(csv_pname, df, summary_columns)

    if wrote_mode == 'rewritten':
        print(
            'Existing summary CSV columns differ from current summary schema; '
            'rewrote file with aligned columns to prevent shifted values.'
        )

    if wrote_mode == 'appended':
        print(f'Summary appended to: {csv_pname}')
    elif wrote_mode == 'rewritten':
        print(f'Summary rewritten with aligned columns: {csv_pname}')
    else:
        print(f'Summary saved to: {csv_pname}')

    print(f'\n Step *{STEP}* completed successfully')
