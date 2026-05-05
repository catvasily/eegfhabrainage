"""
**Use trained model to classify new records.**
"""
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from do_xgboost import (
    apply_fitted_reducer,
    list_src_ids,
    load_data,
    load_labels,
    load_xgb_model_from_ubj_buffer,
    verify_pickle_xgboost_version,
)
from plot_pr_curve import plot_and_save_pr_curve

STEP = 'predict'


def cls_predict(ss):
    """
    Load a previously saved trained model and use it to predict labels
    for a set of new CWT amp distribution records.

    The model pickle is identified via ``predict.model`` settings in the JSON:

    - If ``predict.model.pickle`` is non-null it is used as a path relative to
      ``out_root``.
    - Otherwise the pickle filename is derived from the other ``predict.model``
      parameters using the same naming convention as the ``xgboost`` step.

    Records to classify are specified under ``predict.input_records`` and follow
    the same selection logic as the ``xgboost`` step (hospital list, optional
    physician filter, optional explicit scan_ids list).

    Outputs:

    - Per-scan CSV with true label, predicted label and class probabilities.
    - Console summary: accuracy, classification report, confusion matrix, PR-AUC.
    - Precision-Recall curve PNG saved to ``out_root``.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    predict_cfg = ss.args.get('predict', {}) or {}

    # ------------------------------------------------------------------ #
    # 1. Resolve & load the trained model pickle                           #
    # ------------------------------------------------------------------ #
    pkl_pname = _resolve_model_pickle_path(ss, predict_cfg)
    print(f'Loading trained model from: {pkl_pname}')

    if not pkl_pname.exists():
        raise FileNotFoundError(f'Model pickle not found: {pkl_pname}')

    with open(pkl_pname, 'rb') as fp:
        payload = pickle.load(fp)
    
    # Most times the version will differ between local machine and cluster
    verify_pickle_xgboost_version(payload, pkl_pname)

    full_model_data = payload.get('full_model_data')

    if full_model_data is None:
        raise KeyError(f'No "full_model_data" key found in pickle: {pkl_pname}')

    # Read saved full model settings for label, use moments, etc
    model_target_label = payload.get('target_label', ss.args.get('target_label', ''))
    model_use_moments_only = bool(payload.get('use_moments_only', True))
    model_ignore_confidence = bool(payload.get('ignore_confidence', False))

    # ------------------------------------------------------------------ #
    # 2. Load input records                                                #
    # ------------------------------------------------------------------ #
    input_records_cfg = predict_cfg.get('input_records', {}) or {}

    # Temporarily override the relevant ss.args keys so that list_src_ids /
    # load_labels / load_data operate on the predict-step settings rather than
    # the JSON global settings.
    override_keys = {
        'hospital': input_records_cfg.get('hospital', ss.args.get('hospital')),
        'physician': input_records_cfg.get('physician', []),
        'scan_ids': input_records_cfg.get('scan_ids', None),
        'target_label': model_target_label,
        'use_moments_only': model_use_moments_only,
        'ignore_confidence': model_ignore_confidence,
    }

    # Save a copy of global settings that will be modified
    saved_args = {k: copy.deepcopy(ss.args.get(k)) for k in override_keys}

    # Install model settings from the pickle as global
    ss.args.update(override_keys)

    try:
        lst_IDs = ss.args['scan_ids']

        if lst_IDs is None:
            lst_IDs = list_src_ids(ss)
            print(f'# of EEG records found for prediction: {len(lst_IDs)}')
        else:
            if len(ss.args['hospital']) > 1:
                raise ValueError(
                    'When scan_ids are explicitly listed, only one hospital should be specified '
                    'in predict.input_records.hospital'
                )

            lst_IDs = [(0, sid) for sid in lst_IDs]

        ss.args['scan_ids'] = lst_IDs

        # load_labels updates ss.args['scan_ids'] in-place (drops unlabeled records)
        Y = load_labels(ss)
        print(f'Labeled records available for prediction: {len(Y)}')

        # Capture scan IDs and hospital list before they could be further modified
        scan_ids_for_output = list(ss.args['scan_ids'])
        hospital_ref = list(override_keys['hospital'])

        raw_features, ch_names, tfd_freqs, tfd_parm_names = load_data(ss)
    finally:
        # Always restore original args, even on error
        for k, v in saved_args.items():
            ss.args[k] = v

    # Drop unknown-lobe channels (mirrors the xgboost step)
    drop_channels = {'Unknown-lh', 'Unknown-rh'}
    keep_idx = [i for i, name in enumerate(ch_names) if name not in drop_channels]
    raw_features = raw_features[:, keep_idx, :, :]
    ch_names = [ch_names[i] for i in keep_idx]

    # ------------------------------------------------------------------ #
    # 3. Apply fitted reducer and run predictions                          #
    # ------------------------------------------------------------------ #
    reducer = full_model_data['reducer']
    feature_names = full_model_data.get('feature_names')
    model = load_xgb_model_from_ubj_buffer(full_model_data['model_ubj'])

    # Temporarily expose ch/freq/parm metadata in ss.args in case the reducer
    # needs them (e.g. consensus_cv, to_lobes).  Restore afterwards.
    ss.args['ch_names'] = ch_names
    ss.args['freqs'] = tfd_freqs
    ss.args['parm_names'] = tfd_parm_names

    X = apply_fitted_reducer(raw_features, reducer)

    if feature_names is not None and len(feature_names) == X.shape[1]:
        X_in = pd.DataFrame(X, columns=feature_names, copy=False)
    else:
        X_in = X

    y_proba = model.predict_proba(X_in)
    y_pred = np.argmax(y_proba, axis=1)
    y_score = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()

    # ------------------------------------------------------------------ #
    # 4. Build and save per-scan results CSV                               #
    # ------------------------------------------------------------------ #
    scan_id_list = [tpl[1] for tpl in scan_ids_for_output]
    hosp_list = [
        hospital_ref[tpl[0]]
        if isinstance(tpl[0], int) and tpl[0] < len(hospital_ref)
        else 'Unknown'
        for tpl in scan_ids_for_output
    ]

    results: dict = {
        'ScanID': scan_id_list,
        'Hospital': hosp_list,
        'TrueLabel': Y,
        'PredLabel': y_pred,
    }

    # Include one column per class probability
    for cls_idx in range(y_proba.shape[1]):
        results[f'Proba_{cls_idx}'] = y_proba[:, cls_idx]

    results_df = pd.DataFrame(results)

    csv_pname = _build_predict_csv_path(ss, pkl_pname, input_records_cfg)
    ss.out_root.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_pname, index=False)
    print(f'\nPer-scan predictions saved to: {csv_pname}')

    # ------------------------------------------------------------------ #
    # 5. Summary classification stats                                      #
    # ------------------------------------------------------------------ #
    print('\n--- Classification Report ---')
    print(f'Accuracy: {accuracy_score(Y, y_pred):.4f}  (n={len(Y)})')
    print(classification_report(Y, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(Y, y_pred))

    if np.unique(Y).size == 2:
        pr_auc = average_precision_score(Y, y_score)
        print(f'\nPR AUC (average precision): {pr_auc:.3f}')

    # ------------------------------------------------------------------ #
    # 6. Precision-Recall curve (binary only)                              #
    # ------------------------------------------------------------------ #
    if np.unique(Y).size == 2:
        show_plot = ss.args.get('show_plots', True)
        prcrv_pname = ss.out_root / f'prcrv_predict_{pkl_pname.stem}.png'
        prcrv_pname = plot_and_save_pr_curve(
            Y,
            y_score,
            step_name=STEP,
            outfname=prcrv_pname,
            show_plot=show_plot,
            target_label=model_target_label,
            physician=input_records_cfg.get('physician'),
        )
        print(f'Precision-Recall curve saved to: {prcrv_pname}')

    print(f'\n Step *{STEP}* completed successfully')


def _build_predict_csv_path(ss, pkl_pname, input_records_cfg):
    """
    Build output CSV path for predict results.

    Naming rules:
    - If ``scan_ids`` is explicitly provided, omit the ``For...`` segment.
    - Otherwise include ``For`` and append ``_{hlist}`` iff hospital list exists.
    - Append ``_{physicians}`` iff physician list exists, joined by ``_``.

    Args:
        ss(obj): reference to this app object
        pkl_pname(Path): source model pickle path
        input_records_cfg(dict): ``predict.input_records`` configuration

    Returns:
        Path: full output CSV path under ``ss.out_root``
    """
    input_hospitals = input_records_cfg.get('hospital', []) or []
    input_physicians = input_records_cfg.get('physician', []) or []
    input_scan_ids = input_records_cfg.get('scan_ids', None)

    if input_scan_ids is not None:
        csv_fname = f'predict_{pkl_pname.stem}.csv'
    else:
        for_suffix = 'For'
        if len(input_hospitals) > 0:
            for_suffix += f'_{ss.hlist(input_hospitals)}'
        if len(input_physicians) > 0:
            for_suffix += f"_{'_'.join(input_physicians)}"
        csv_fname = f'predict_{pkl_pname.stem}{for_suffix}.csv'

    return ss.out_root / csv_fname


def _resolve_model_pickle_path(ss, predict_cfg):
    """
    Resolve the fully-qualified path to the trained model pickle.

    If ``predict.model.pickle`` is non-null it is treated as a path relative
    to ``out_root``.  Otherwise the name is derived from the model parameters
    using the same naming convention as the ``xgboost`` step, by temporarily
    overriding the relevant ``ss.args`` keys so that ``ss.cls_pkl_pname``
    generates the correct name.

    Args:
        ss(obj): reference to this app object
        predict_cfg(dict): the ``predict`` section of the input JSON

    Returns:
        pkl_pname(Path): fully-resolved path to the pickle file

    """
    model_cfg = predict_cfg.get('model', {}) or {}
    explicit_pickle = model_cfg.get('pickle')

    if explicit_pickle is not None:
        return ss.out_root / explicit_pickle

    hospital = model_cfg.get('hospital')
    label = model_cfg.get('target_label')
    use_moments_only = bool(model_cfg.get('use_moments_only', True))
    standardize = bool(model_cfg.get('standardize_features', False))
    ignore_confidence = bool(model_cfg.get('ignore_confidence', False))

    if not hospital:
        raise ValueError(
            '"predict.model.hospital" must be specified when "predict.model.pickle" is null'
        )
    if not label:
        raise ValueError(
            '"predict.model.target_label" must be specified when "predict.model.pickle" is null'
        )

    nparms = 5 if use_moments_only else 9
    hlist = ss.hlist(hospital)

    # Temporarily override reduction-related args so that the internal
    # _reducer_pickle_tag() helper inside ss.cls_pkl_pname returns the right
    # tag for the model we're looking for.
    tag_keys = ('dim_reduction', 'use_consensus_cv')
    saved = {k: copy.deepcopy(ss.args.get(k)) for k in tag_keys}

    if 'dim_reduction' in model_cfg:
        ss.args['dim_reduction'] = model_cfg['dim_reduction']
    if 'use_consensus_cv' in model_cfg:
        ss.args['use_consensus_cv'] = model_cfg['use_consensus_cv']

    try:
        pkl_pname = ss.cls_pkl_pname(hlist, label, nparms, standardize, ignore_confidence)
    finally:
        for k, v in saved.items():
            ss.args[k] = v

    return pkl_pname
