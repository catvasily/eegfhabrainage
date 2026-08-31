"""
**Use trained model to classify new records.**
"""
import copy
import pickle
import re
from pathlib import Path

import cls_calibrate
import numpy as np
import pandas as pd
from scipy.stats import brunnermunzel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from cls_summary_utils import (
    align_summary_dataframe,
    append_or_merge_summary_csv,
    build_summary_row,
    get_summary_columns,
    physician_token,
)
from effect_size import EffectSizeCalculator
from auc_bootstrap_stats import cohens_d_from_stats, stratified_bootstrap_metrics
from cls_calibrate import (
    SUPPORTED_CALIBRATION_METHODS,
    SUPPORTED_CALIBRATION_METHODS_WITH_BEST,
)
from do_xgboost import (
    apply_fitted_reducer,
    list_src_ids,
    load_data,
    load_labels,
    load_xgb_model_from_ubj_buffer,
    verify_pickle_xgboost_version,
)
from joint_predictor import JointBayes, JointLR
from plot_pr_curve import plot_and_save_pr_curve

STEP = 'predict'


def cls_predict(ss):
    """
    Load a previously saved trained model and use it to predict labels
    for a set of new CWT amp distribution records.

    The model pickle is identified via ``predict.model`` settings in the JSON:

        - If top-level ``pickle`` is non-null it is used as a path relative to
      ``out_root``.
        - Otherwise the pickle filename is derived from ``predict.model``
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
    summarize_only = bool(predict_cfg.get('summarize_only', False))
    plot_only = bool(predict_cfg.get('plot_only', False))

    if summarize_only and plot_only:
        raise ValueError(
            'predict.summarize_only and predict.plot_only are mutually exclusive. '
            'Set exactly one of them to true (or both false for normal prediction mode).'
        )

    if plot_only:
        summarize_cfg = predict_cfg.get('summarize', {}) or {}

        if not summarize_cfg:
            raise ValueError('predict.summarize must be configured when predict.plot_only is true.')

        plot_outputs = _run_predict_plot_only(ss, predict_cfg)

        if isinstance(plot_outputs, dict):
            ridge_png = plot_outputs.get('ridge_plot')
            counts_png = plot_outputs.get('counts_bar_plot')
            aucs_png = plot_outputs.get('aucs_bar_plot')
            grid_pr_png = plot_outputs.get('grid_pr_plot')
            mcc_png = plot_outputs.get('mcc_error_plot')

            if ridge_png is not None:
                print(f'Predict ridge-plot grid saved to: {ridge_png}')

            if counts_png is not None:
                print(f'Predict counts bar plot saved to: {counts_png}')

            if aucs_png is not None:
                print(f'Predict PR AUC bar plot saved to: {aucs_png}')

            if grid_pr_png is not None:
                print(f'Predict grid PR plot saved to: {grid_pr_png}')

            if mcc_png is not None:
                print(f'Predict error-MCC heatmap saved to: {mcc_png}')
        else:
            print(f'Predict ridge-plot grid saved to: {plot_outputs}')

        print(f'\n Step *{STEP}* completed successfully')
        return

    if summarize_only:
        _summarize_predict_outputs(ss, predict_cfg)
        print(f'\n Step *{STEP}* completed successfully')
        return

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
    model_physician = payload.get('physician', ss.args.get('physician', []))

    # ------------------------------------------------------------------ #
    # 2. Load input records                                                #
    # ------------------------------------------------------------------ #
    input_records_cfg = predict_cfg.get('input_records', {}) or {}
    target_physician_cfg = input_records_cfg.get('physician', None)
    input_physician = input_records_cfg.get('physician', None)

    if input_physician is None:
        input_physician = model_physician

    input_ignore_confidence = input_records_cfg.get('ignore_confidence', None)

    if input_ignore_confidence is None:
        input_ignore_confidence = model_ignore_confidence
    else:
        input_ignore_confidence = bool(input_ignore_confidence)

    # Temporarily override the relevant ss.args keys so that list_src_ids /
    # load_labels / load_data operate on the predict-step settings rather than
    # the JSON global settings.
    override_keys = {
        'hospital': input_records_cfg.get('hospital', ss.args.get('hospital')),
        'physician': input_physician,
        'scan_ids': input_records_cfg.get('scan_ids', None),
        'target_label': model_target_label,
        'use_moments_only': model_use_moments_only,
        'ignore_confidence': input_ignore_confidence,
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

    use_calibrated_p = bool(predict_cfg.get('use_calibrated_p', True))
    requested_calibration_method = _resolve_predict_calibration_method(predict_cfg)

    # ------------------------------------------------------------------ #
    # 3. Apply fitted reducer and run predictions                          #
    # ------------------------------------------------------------------ #
    reducer = full_model_data['reducer']
    feature_names = full_model_data.get('feature_names')
    model = load_xgb_model_from_ubj_buffer(full_model_data['model_ubj'])
    _configure_predict_model_threads(model, (predict_cfg.get('model', {}) or {}))

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

    y_proba_raw = model.predict_proba(X_in)

    if y_proba_raw.ndim != 2 or y_proba_raw.shape[1] < 2:
        raise ValueError(
            'Prediction requires binary class probabilities with shape (n_samples, 2).'
        )

    y_score_raw = y_proba_raw[:, 1]

    threshold_model_y_true = payload.get('Y', None)
    threshold_model_y_score = payload.get('y_score', None)

    calibration_method = None

    if use_calibrated_p:
        calibration_meta = _ensure_payload_calibration(
            payload=payload,
            pkl_pname=pkl_pname,
            calibrate_cfg=(ss.args.get('calibrate', {}) or {}),
            requested_method=requested_calibration_method,
        )
        calibration_method = calibration_meta.get('method', None)
        y_score = _apply_calibrator_to_raw_scores(y_score_raw, calibration_meta)
        y_proba = np.column_stack((1.0 - y_score, y_score))

        # Keep threshold optimization consistent with calibrated inference by
        # reusing calibrated OOF training scores from this same pickle.
        threshold_model_y_true, model_score_raw = _extract_binary_scores_from_payload(payload, pkl_pname)
        threshold_model_y_score = _apply_calibrator_to_raw_scores(model_score_raw, calibration_meta)

        print(f'Using calibrated probabilities via method: {calibration_meta["method"]}')
    else:
        y_proba = y_proba_raw
        y_score = y_score_raw

    y_pred, y_confidence, decision_threshold, threshold_method = _predict_labels_with_thresholding(
        y_proba=y_proba,
        y_true=Y,
        model_y_true=threshold_model_y_true,
        model_y_score=threshold_model_y_score,
        pkl_pname=pkl_pname,
        threshold_selection=(predict_cfg.get('model', {}) or {}).get('threshold_selection', None),
    )
    print(
        f'Predict threshold selection: {threshold_method}; '
        f'decision threshold={decision_threshold:.6f}'
    )

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
        'PredConfidence': y_confidence,
        'DecisionThreshold': decision_threshold,
        'ThresholdSelection': threshold_method,
    }

    # Include one column per class probability
    for cls_idx in range(y_proba.shape[1]):
        results[f'Proba_{cls_idx}'] = y_proba[:, cls_idx]

    results_df = pd.DataFrame(results)

    physician_pair = _build_physician_pair(
        model_physician,
        target_physician_cfg,
        input_ignore_confidence,
    )
    csv_pname = _build_predict_csv_path(
        ss,
        pkl_pname,
        physician_pair,
        calibration_method=calibration_method,
    )
    _predict_results_root(ss).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_pname, index=False)
    print(f'\nPer-scan predictions saved to: {csv_pname}')

    _append_predict_summary_row(
        ss=ss,
        payload=payload,
        pkl_pname=pkl_pname,
        Y=Y,
        y_pred=y_pred,
        y_score=y_score,
        physician_pair=physician_pair,
        calibration_method=calibration_method,
    )

    # ------------------------------------------------------------------ #
    # 5. Summary classification stats                                      #
    # ------------------------------------------------------------------ #
    print('\n--- Classification Report ---')
    print(f'Accuracy: {accuracy_score(Y, y_pred):.4f}  (n={len(Y)})')
    print(classification_report(Y, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(Y, y_pred))

    bs_value = np.nan

    if np.unique(Y).size == 2:
        pr_auc = average_precision_score(Y, y_score)
        bs_value = brier_score_loss(Y, y_score)
        print(f'\nPR AUC (average precision): {pr_auc:.3f}')
        print(f'Brier score: {bs_value:.3f}')

    # ------------------------------------------------------------------ #
    # 6. Precision-Recall curve (binary only)                              #
    # ------------------------------------------------------------------ #
    if np.unique(Y).size == 2:
        show_plot = ss.args.get('show_plots', True)
        prcrv_pname = _build_predict_prcurve_png_path(
            ss,
            pkl_pname,
            physician_pair,
            calibration_method=calibration_method,
        )
        prcrv_pname.parent.mkdir(parents=True, exist_ok=True)
        prcrv_pname = plot_and_save_pr_curve(
            Y,
            y_score,
            step_name=STEP,
            outfname=prcrv_pname,
            show_plot=show_plot,
            target_label=model_target_label,
            physician=physician_pair,
            brier_score=bs_value,
        )
        print(f'Precision-Recall curve saved to: {prcrv_pname}')

    print(f'\n Step *{STEP}* completed successfully')


def _build_predict_csv_path(ss, pkl_pname, physician_pair, calibration_method=None):
    """
    Build output CSV path for predict per-scan labels.

    Naming rule: start with model pickle name, replace ``xgb`` prefix with
    ``<model-physician>2<target-physician>``, and drop everything between
    ``ignoreConf<True|False>`` and ``.pkl``.

    Args:
        ss(obj): reference to this app object
        pkl_pname(Path): source model pickle path
        physician_pair(str): ``<model-physician>2<target-physician>`` token

    Returns:
        Path: full output CSV path under ``ss.out_root``
    """
    base_name = _build_predict_output_base_name(pkl_pname, physician_pair)
    suffix = _predict_calibration_file_suffix(calibration_method)
    return _predict_results_root(ss) / f'{base_name}{suffix}.csv'


def _run_predict_plot_only(ss, predict_cfg):
    """Load plotting helper module and create predict plot-only visualizations."""
    import importlib.util

    module_path = Path(__file__).with_name('plt_predict.py')
    spec = importlib.util.spec_from_file_location('plt_predict', module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to prepare import spec for plotting module: {module_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'plot_predict_outputs'):
        return module.plot_predict_outputs(ss, predict_cfg)

    return {
        'ridge_plot': module.plot_predict_ridge_grid(ss, predict_cfg),
    }


def _build_predict_prcurve_png_path(ss, pkl_pname, physician_pair, calibration_method=None):
    """
    Build PR curve output PNG path using the exact same basename as predict CSV,
    with only ``prcrv_`` prefix and ``.png`` extension difference.
    """
    base_name = _build_predict_output_base_name(pkl_pname, physician_pair)
    suffix = _predict_calibration_file_suffix(calibration_method)
    return _predict_results_root(ss) / 'plt' / f'prcrv_{base_name}{suffix}.png'


def _predict_calibration_file_suffix(calibration_method):
    """Return filename suffix for calibrated predict outputs."""
    if calibration_method is None:
        return ''

    method = str(calibration_method).strip().lower()

    if not method:
        return ''

    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f'Unsupported calibration method for filename suffix: {calibration_method!r}. '
            f'Expected one of: {SUPPORTED_CALIBRATION_METHODS!r}.'
        )

    return f'_{method}'


def _resolve_predict_calibration_method(predict_cfg):
    """Return normalized calibration method requested by predict config, or None."""
    use_calibrated_p = bool((predict_cfg or {}).get('use_calibrated_p', True))

    if not use_calibrated_p:
        return None

    requested_method = str((predict_cfg or {}).get('calibration', 'betacal')).strip().lower()

    if requested_method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f'predict.calibration must be one of: {SUPPORTED_CALIBRATION_METHODS!r}.'
        )

    return requested_method


def _platt_defaults():
    """Default constructor settings for sklearn LogisticRegression used in Platt scaling."""
    return {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 1.0,
        'fit_intercept': True,
        'intercept_scaling': 1.0,
        'class_weight': None,
        'random_state': None,
        'solver': 'lbfgs',
        'max_iter': 100,
        'multi_class': 'auto',
        'verbose': 0,
        'warm_start': False,
        'n_jobs': None,
        'l1_ratio': None,
    }


def _betacal_defaults():
    """Default constructor settings for betacal.BetaCalibration."""
    return {
        'parameters': 'abm',
    }


def _isotonic_defaults():
    """Default constructor settings for sklearn IsotonicRegression."""
    return {
        'y_min': 0.0,
        'y_max': 1.0,
        'increasing': True,
        'out_of_bounds': 'clip',
    }


def _extract_binary_scores_from_payload(payload, pkl_pname):
    """Read binary calibration arrays from payload and validate basic assumptions."""
    if 'Y' not in payload:
        raise KeyError(f'No "Y" key found in pickle: {pkl_pname}')

    if 'y_proba' not in payload:
        raise KeyError(f'No "y_proba" key found in pickle: {pkl_pname}')

    y_true = np.asarray(payload['Y']).ravel()
    y_proba = np.asarray(payload['y_proba'])

    if y_true.size == 0:
        raise ValueError('Saved true labels array "Y" is empty.')

    if y_proba.ndim == 1:
        y_score = y_proba
    elif y_proba.ndim == 2:
        if y_proba.shape[1] < 2:
            raise ValueError(
                'Saved "y_proba" must have at least 2 columns for binary classes '
                'when provided as a 2D array.'
            )
        y_score = y_proba[:, 1]
    else:
        raise ValueError('Saved "y_proba" must be either a 1D or 2D array.')

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            'Saved "Y" and "y_proba" length mismatch: '
            f'n_true={y_true.shape[0]}, n_proba={y_score.shape[0]}'
        )

    if np.any(~np.isfinite(y_score)):
        raise ValueError('Saved "y_proba" contains non-finite values.')

    if np.any((y_score < 0.0) | (y_score > 1.0)):
        raise ValueError('Saved "y_proba" values must be within [0, 1].')

    if np.any(~np.isin(y_true, [0, 1])):
        raise ValueError('Saved "Y" must contain binary labels encoded as 0/1.')

    return y_true, y_score


def _fit_payload_calibrator(y_true, y_score, calibrate_cfg):
    """Fit a calibrator from payload OOF probabilities following calibrate.* settings."""
    method = str(calibrate_cfg.get('method', 'platt')).strip().lower()

    if not method:
        raise ValueError('calibrate.method cannot be empty.')

    def _fit_one(method_name, strict=True):
        if method_name == 'platt':
            platt_cfg = _platt_defaults()
            platt_cfg.update(calibrate_cfg.get('platt', {}) or {})

            platt = LogisticRegression(**platt_cfg)
            x_score = y_score.reshape(-1, 1)
            platt.fit(x_score, y_true)
            return platt, platt.predict_proba(x_score)[:, 1]

        if method_name == 'betacal':
            try:
                from betacal import BetaCalibration
            except ImportError as ex:
                if strict:
                    raise ImportError(
                        'calibrate.method="betacal" requires the "betacal" package. '
                        'Install it with: pip install betacal'
                    ) from ex
                return None, None

            betacal_cfg = _betacal_defaults()
            betacal_cfg.update(calibrate_cfg.get('betacal', {}) or {})

            beta = BetaCalibration(**betacal_cfg)
            beta.fit(y_score, y_true)
            return beta, np.asarray(beta.predict(y_score)).ravel()

        if method_name == 'isotonic':
            isotonic_cfg = _isotonic_defaults()
            isotonic_cfg.update(calibrate_cfg.get('isotonic', {}) or {})

            iso = IsotonicRegression(**isotonic_cfg)
            iso.fit(y_score, y_true)
            return iso, np.asarray(iso.predict(y_score)).ravel()

        raise ValueError(
            f'Unsupported calibration method={method_name!r}. '
            f'Supported methods: {SUPPORTED_CALIBRATION_METHODS_WITH_BEST!r}.'
        )

    if method == 'best':
        candidate_methods = SUPPORTED_CALIBRATION_METHODS
        best_method = None
        best_score = None
        best_y_score = None
        best_fitted = None

        print('Evaluating calibration methods for calibrate.method="best":')
        for candidate in candidate_methods:
            fitted_c, y_candidate = _fit_one(candidate, strict=False)

            if y_candidate is None:
                print(f'  {candidate}: skipped (dependency not available)')
                continue

            candidate_score = brier_score_loss(y_true, y_candidate)
            print(f'  {candidate}: corrected Brier score = {candidate_score:.4e}')

            if (best_score is None) or (candidate_score < best_score):
                best_method = candidate
                best_score = candidate_score
                best_y_score = y_candidate
                best_fitted = fitted_c

        if best_method is None:
            raise RuntimeError('No calibration methods could be evaluated for method="best".')

        print(f'Selected best calibration method: {best_method} (Brier={best_score:.4e})')
        return best_method, best_fitted, best_y_score

    fitted, y_score_corrected = _fit_one(method, strict=True)
    return method, fitted, y_score_corrected


def _apply_calibrator_to_raw_scores(y_score_raw, calibration_meta):
    """Apply stored calibrator to raw model positive-class scores."""
    method = calibration_meta['method']
    calibrator = calibration_meta['calibrator']
    y_score_raw = np.asarray(y_score_raw, dtype=float).ravel()

    if method == 'platt':
        y_score_cal = calibrator.predict_proba(y_score_raw.reshape(-1, 1))[:, 1]
    elif method in ('betacal', 'isotonic'):
        y_score_cal = np.asarray(calibrator.predict(y_score_raw)).ravel()
    else:
        raise ValueError(
            f'Cannot apply stored calibration method={method!r}: unknown method.'
        )

    if np.any(~np.isfinite(y_score_cal)):
        raise ValueError('Calibrated probabilities contain non-finite values.')

    if np.any((y_score_cal < 0.0) | (y_score_cal > 1.0)):
        raise ValueError('Calibrated probabilities must be within [0, 1].')

    return y_score_cal


def _ensure_payload_calibration(payload, pkl_pname, calibrate_cfg, requested_method):
    """Return calibration metadata, fitting/saving it first when missing/mismatched/forced."""
    force_recalibrate = bool(calibrate_cfg.get('force_recalibrate', False))
    stored = payload.get('calibration')
    requested_method = str(requested_method).strip().lower()

    if requested_method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f'predict.calibration must be one of: {SUPPORTED_CALIBRATION_METHODS!r}.'
        )

    stored_method = None
    if isinstance(stored, dict):
        stored_method = str(stored.get('method', '')).strip().lower()

    method_matches = bool(stored is not None) and (stored_method == requested_method)

    if method_matches and not force_recalibrate:
        return stored

    if stored is None:
        print(
            f'Calibration missing in pickle; fitting requested method={requested_method!r}.'
        )
    elif not method_matches:
        print(
            'Stored calibration method does not match requested method; '
            f'stored={stored_method!r}, requested={requested_method!r}. Recalibrating.'
        )
    elif force_recalibrate:
        print(
            'calibrate.force_recalibrate=true; '
            f'refitting calibration with method={requested_method!r}.'
        )

    y_true, y_score = _extract_binary_scores_from_payload(payload, pkl_pname)
    calibrate_cfg_for_fit = dict(calibrate_cfg)
    calibrate_cfg_for_fit['method'] = requested_method
    method, fitted_calibrator, y_score_corrected = _fit_payload_calibrator(
        y_true,
        y_score,
        calibrate_cfg_for_fit,
    )

    brier_uncalibrated = brier_score_loss(y_true, y_score)
    brier_corrected = brier_score_loss(y_true, y_score_corrected)

    payload['calibration'] = {
        'method': method,
        'calibrator': fitted_calibrator,
        'brier_uncalibrated': brier_uncalibrated,
        'brier_corrected': brier_corrected,
    }

    with open(pkl_pname, 'wb') as fp:
        pickle.dump(payload, fp)

    print(f'Calibrator ({method}) saved to pickle: {pkl_pname}')
    return payload['calibration']


def _build_predict_output_base_name(pkl_pname, physician_pair):
    """
    Build shared basename for predict outputs from model pickle name.
    """
    pkl_name = pkl_pname.name

    if pkl_name.startswith('xgb'):
        out_name = physician_pair + pkl_name[3:]
    else:
        out_name = f'{physician_pair}_{pkl_name}'

    out_name = re.sub(r'(ignoreConf(?:True|False)).*\.pkl$', r'\1', out_name)

    if out_name.endswith('.pkl'):
        out_name = Path(out_name).stem

    return out_name


def _append_predict_summary_row(
    ss,
    payload,
    pkl_pname,
    Y,
    y_pred,
    y_score,
    physician_pair,
    calibration_method=None,
):
    """
    Append one prediction-results summary row to the common summary CSV.

    All writes go through ``append_or_merge_summary_csv`` in
    ``cls_summary_utils``, which holds a cross-process advisory lock
    (``fcntl.LOCK_EX``) for the duration of each read-modify-write cycle.
    Any other code that writes to the same summary CSV must also use that
    helper to remain concurrency-safe.
    """
    summary_columns = get_summary_columns(ss)
    row = build_summary_row(
        ss=ss,
        payload=payload,
        pkl_pname=pkl_pname,
        y_true=Y,
        y_pred=y_pred,
        y_score=y_score,
        physician_cell=physician_pair,
        calibration=calibration_method or 'none',
    )
    df = align_summary_dataframe(pd.DataFrame([row]), summary_columns)

    summary_csv_name = ss.args['summary_csv_name']
    csv_pname = _predict_results_root(ss) / summary_csv_name
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


def _summarize_predict_outputs(ss, predict_cfg):
    """
    Summarize existing predict CSV files across multiple model physicians.

    This mode does not run any model predictions. It only reads existing CSV
    files produced by this module and builds one aggregated per-scan table.
    """
    summarize_cfg = predict_cfg.get('summarize', {}) or {}

    if not summarize_cfg:
        raise ValueError('predict.summarize must be configured when predict.summarize_only is true.')

    model_physicians = _normalize_physician_list(
        summarize_cfg.get('model_physicians', None),
        'predict.summarize.model_physicians',
    )

    if not model_physicians:
        raise ValueError('predict.summarize.model_physicians must contain at least one physician.')

    target_physicians = summarize_cfg.get('target_physicians', None)

    model_cfg = _build_summarize_model_cfg(ss, predict_cfg, summarize_cfg)
    model_pickle_path = _build_model_pickle_path_from_model_cfg(ss, model_cfg)

    # In the code below, 'all_' prefixed variables refer to a full set of
    # records where both confident and non-confident records are pooled together.

    all_scan_ids_by_physician = {}
    all_predictions_by_physician = {}
    all_confidence_by_physician = {}
    conf_scan_ids_by_physician = {}
    true_label_by_scan = {}
    threshold_selection_by_csv = {}
    decision_threshold_by_physician = {}

    for physician_name in model_physicians:
        physician_one = [physician_name]

        all_pair = _build_physician_pair(physician_one, target_physicians, True)
        conf_pair = _build_physician_pair(physician_one, target_physicians, False)

        calibration_method = _resolve_predict_calibration_method(predict_cfg)

        all_csv = _build_predict_csv_path(
            ss,
            model_pickle_path,
            all_pair,
            calibration_method=calibration_method,
        )
        conf_csv = _build_predict_csv_path(
            ss,
            model_pickle_path,
            conf_pair,
            calibration_method=calibration_method,
        )

        if not all_csv.exists():
            raise FileNotFoundError(f'Expected All-level predict CSV not found: {all_csv}')

        if not conf_csv.exists():
            raise FileNotFoundError(f'Expected Conf-level predict CSV not found: {conf_csv}')

        df_all = _load_predict_results_csv(all_csv)
        df_conf = _load_predict_results_csv(conf_csv)

        # Temporary legacy correction: detect and fix old threshold-normalized confidence
        # (which used [-1, 1] range with negative values).
        # In future, this branch will not be needed once all legacy CSVs are regenerated.
        if 'PredConfidence' in df_all.columns:
            old_conf = pd.to_numeric(df_all['PredConfidence'], errors='coerce')
            if np.any(old_conf < 0.0):
                print(
                    f'Detected legacy threshold-normalized confidence (negative values) in {all_csv}. '
                    'Recalculating using new predicted-label probability method and saving corrected CSV.'
                )
                if not {'PredLabel', 'Proba_1'}.issubset(df_all.columns):
                    raise ValueError(
                        f'Cannot correct legacy confidence in {all_csv}: '
                        'missing PredLabel or Proba_1 columns required for recalculation.'
                    )
                pred_vals = pd.to_numeric(df_all['PredLabel'], errors='coerce')
                proba_pos_vals = pd.to_numeric(df_all['Proba_1'], errors='coerce')
                new_conf = _predicted_label_probability_confidence(proba_pos_vals, pred_vals)
                df_all['PredConfidence'] = new_conf
                df_all.to_csv(all_csv, index=False)
        
        if 'PredConfidence' in df_conf.columns:
            old_conf = pd.to_numeric(df_conf['PredConfidence'], errors='coerce')
            if np.any(old_conf < 0.0):
                print(
                    f'Detected legacy threshold-normalized confidence (negative values) in {conf_csv}. '
                    'Recalculating using new predicted-label probability method and saving corrected CSV.'
                )
                if not {'PredLabel', 'Proba_1'}.issubset(df_conf.columns):
                    raise ValueError(
                        f'Cannot correct legacy confidence in {conf_csv}: '
                        'missing PredLabel or Proba_1 columns required for recalculation.'
                    )
                pred_vals = pd.to_numeric(df_conf['PredLabel'], errors='coerce')
                proba_pos_vals = pd.to_numeric(df_conf['Proba_1'], errors='coerce')
                new_conf = _predicted_label_probability_confidence(proba_pos_vals, pred_vals)
                df_conf['PredConfidence'] = new_conf
                df_conf.to_csv(conf_csv, index=False)

        threshold_selection_by_csv[str(all_csv)] = _extract_csv_threshold_selection(df_all, all_csv)
        threshold_selection_by_csv[str(conf_csv)] = _extract_csv_threshold_selection(df_conf, conf_csv)
        decision_threshold_by_physician[physician_name] = _extract_csv_decision_threshold(df_all, all_csv)

        _merge_true_labels(true_label_by_scan, df_all, all_csv)
        _merge_true_labels(true_label_by_scan, df_conf, conf_csv)

        all_scan_ids_by_physician[physician_name] = list(df_all['ScanID'])
        all_predictions_by_physician[physician_name] = dict(
            zip(df_all['ScanID'], pd.to_numeric(df_all['PredLabel'], errors='coerce'))
        )

        if not {'PredLabel', 'Proba_1'}.issubset(df_all.columns):
            raise ValueError(
                f'CSV file {all_csv} must contain PredLabel and Proba_1 columns '
                'for summarize-only confidence computation.'
            )

        pred_vals = pd.to_numeric(df_all['PredLabel'], errors='coerce')
        proba_pos_vals = pd.to_numeric(df_all['Proba_1'], errors='coerce')
        conf_vals = _predicted_label_probability_confidence(proba_pos_vals, pred_vals)
        conf_vals = pd.Series(conf_vals, index=df_all.index, dtype=float)

        all_confidence_by_physician[physician_name] = dict(
            zip(df_all['ScanID'], conf_vals)
        )
        conf_scan_ids_by_physician[physician_name] = set(df_conf['ScanID'])

    ordered_scan_ids = []
    seen_scan_ids = set()

    for physician_name in model_physicians:
        for scan_id in all_scan_ids_by_physician[physician_name]:
            if scan_id in seen_scan_ids:
                continue

            seen_scan_ids.add(scan_id)
            ordered_scan_ids.append(scan_id)

    out_df = pd.DataFrame({'ScanID': ordered_scan_ids})
    out_df['TrueLabel'] = [true_label_by_scan.get(scan_id, np.nan) for scan_id in ordered_scan_ids]

    confident_flags = []

    for scan_id in ordered_scan_ids:
        flags = [1 if scan_id in conf_scan_ids_by_physician[name] else 0 for name in model_physicians]

        if len(set(flags)) != 1:
            raise ValueError(
                f'Inconsistent confident status across physicians for scan_id={scan_id}. '
                f'Flags={dict(zip(model_physicians, flags))}'
            )

        confident_flags.append(flags[0])

    out_df['Confident'] = confident_flags

    for physician_name in model_physicians:
        out_df[physician_name] = [
            all_predictions_by_physician[physician_name].get(scan_id, np.nan)
            for scan_id in ordered_scan_ids
        ]
        out_df[f'{physician_name}_PredConfidence'] = [
            all_confidence_by_physician[physician_name].get(scan_id, np.nan)
            for scan_id in ordered_scan_ids
        ]

    out_df['Mean'] = out_df[model_physicians].mean(axis=1, skipna=True)
    out_df['AbsErr'] = (pd.to_numeric(out_df['TrueLabel'], errors='coerce') - out_df['Mean']).abs()
    out_df['STD'] = out_df[model_physicians].std(axis=1, skipna=True, ddof=1)

    confidence_columns = [f'{physician_name}_PredConfidence' for physician_name in model_physicians]
    out_df['MeanConfidence'] = out_df[confidence_columns].mean(axis=1, skipna=True)
    out_df['STDConfidence'] = out_df[confidence_columns].std(axis=1, skipna=True, ddof=1)

    joint_prediction_cfg = summarize_cfg.get('joint_prediction', {}) or {}
    joint_threshold_summary = add_joint_predictions(out_df, joint_prediction_cfg, ss, model_cfg)
    joint_threshold = float(joint_threshold_summary[1])

    bootstrap_n = int(summarize_cfg.get('n_bootstraps', 2000))
    bootstrap_ci = float(summarize_cfg.get('ci_level', 0.95))
    bootstrap_seed = ss.args.get('seed', None)

    if bootstrap_seed is None:
        raise ValueError('Top-level ss.args["seed"] must be set for summarize bootstrap metrics.')

    bootstrap_seed = int(bootstrap_seed)

    if bootstrap_n <= 0:
        raise ValueError('predict.summarize.n_bootstraps must be a positive integer.')

    if not (0.0 < bootstrap_ci < 1.0):
        raise ValueError('predict.summarize.ci_level must be strictly between 0 and 1.')

    bootstrap_alpha = 1.0 - bootstrap_ci

    confident_mask = pd.to_numeric(out_df['Confident'], errors='coerce') == 1
    non_confident_mask = pd.to_numeric(out_df['Confident'], errors='coerce') == 0

    def _brunnermunzel_pvalue(series_confident, series_nonconfident):
        """Compute a two-sample Brunner-Munzel p-value between confidence groups.

        Args:
            series_confident (pd.Series): Values for records with ``Confident == 1``.
            series_nonconfident (pd.Series): Values for records with ``Confident == 0``.

        Returns:
            float: The Brunner-Munzel test p-value. Returns ``np.nan`` if either group
            has fewer than two finite observations after numeric coercion and NaN
            removal.
        """
        vals_confident = pd.to_numeric(series_confident, errors='coerce').dropna().to_numpy(dtype=float)
        vals_nonconfident = pd.to_numeric(series_nonconfident, errors='coerce').dropna().to_numpy(dtype=float)

        if vals_confident.size < 2 or vals_nonconfident.size < 2:
            return np.nan

        test_result = brunnermunzel(
            vals_confident,
            vals_nonconfident,
            nan_policy='omit',
        )
        return float(test_result.pvalue)

    def _cohens_d_effect_stats(series_confident, series_nonconfident):
        """Compute Cohen's d and its confidence interval for two groups.

        Args:
            series_confident (pd.Series): Values for records with ``Confident == 1``.
            series_nonconfident (pd.Series): Values for records with ``Confident == 0``.

        Returns:
            dict | None: A dictionary with keys ``cohens_d``, ``ci_lower``, and
            ``ci_upper`` when computation succeeds, where CI bounds correspond to
            ``bootstrap_ci``. Returns ``None`` if effect statistics cannot be computed.
        """
        vals_confident = pd.to_numeric(series_confident, errors='coerce').to_numpy(dtype=float)
        vals_nonconfident = pd.to_numeric(series_nonconfident, errors='coerce').to_numpy(dtype=float)

        try:
            effect_calc = EffectSizeCalculator(vals_confident, vals_nonconfident, paired=False)
            ci_lower, ci_upper = effect_calc.confidence_interval(
                effect='cohen',
                confidence=bootstrap_ci,
            )
            return {
                'cohens_d': effect_calc.cohens_d(),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }
        except ValueError:
            return None

    def _fmt_cohens_d_ci(effect_stats):
        if not isinstance(effect_stats, dict):
            return np.nan

        d = pd.to_numeric(pd.Series([effect_stats.get('cohens_d', np.nan)]), errors='coerce').iloc[0]
        ci_lower = pd.to_numeric(pd.Series([effect_stats.get('ci_lower', np.nan)]), errors='coerce').iloc[0]
        ci_upper = pd.to_numeric(pd.Series([effect_stats.get('ci_upper', np.nan)]), errors='coerce').iloc[0]

        if pd.isna(d) or pd.isna(ci_lower) or pd.isna(ci_upper):
            return np.nan

        return f'{float(d):.3f} [{float(ci_lower):.3f}, {float(ci_upper):.3f}]'

    if 'JointConfidence' in out_df.columns:
        joint_confident_values = out_df.loc[confident_mask, 'JointConfidence']
        joint_non_confident_values = out_df.loc[non_confident_mask, 'JointConfidence']
    else:
        joint_confident_values = pd.Series(dtype=float)
        joint_non_confident_values = pd.Series(dtype=float)

    first_physician_conf_col = f'{model_physicians[0]}_PredConfidence'
    summary_stats = [
        (
            'mean_confidence_confident',
            pd.to_numeric(out_df.loc[confident_mask, 'MeanConfidence'], errors='coerce').mean(),
        ),
        (
            'mean_confidence_nonconfident',
            pd.to_numeric(out_df.loc[non_confident_mask, 'MeanConfidence'], errors='coerce').mean(),
        ),
        (
            'median_confidence_confident',
            pd.to_numeric(out_df.loc[confident_mask, 'MeanConfidence'], errors='coerce').median(),
        ),
        (
            'median_confidence_nonconfident',
            pd.to_numeric(out_df.loc[non_confident_mask, 'MeanConfidence'], errors='coerce').median(),
        ),
        (
            'median_std_confidence_confident',
            pd.to_numeric(out_df.loc[confident_mask, 'STDConfidence'], errors='coerce').median(),
        ),
        (
            'median_std_confidence_nonconfident',
            pd.to_numeric(out_df.loc[non_confident_mask, 'STDConfidence'], errors='coerce').median(),
        ),
        (
            'cohens_d_confidence_confident_vs_nonconfident',
            _fmt_cohens_d_ci(_cohens_d_effect_stats(
                out_df.loc[confident_mask, 'MeanConfidence'],
                out_df.loc[non_confident_mask, 'MeanConfidence'],
            )),
        ),
        (
            'cohens_d_std_conf_confident_vs_nonconfident',
            _fmt_cohens_d_ci(_cohens_d_effect_stats(
                out_df.loc[confident_mask, 'STDConfidence'],
                out_df.loc[non_confident_mask, 'STDConfidence'],
            )),
        ),
        (
            'pvalue_brunnermunzel_confidence_confident_vs_nonconfident',
            _brunnermunzel_pvalue(
                out_df.loc[confident_mask, 'MeanConfidence'],
                out_df.loc[non_confident_mask, 'MeanConfidence'],
            ),
        ),
        (
            'pvalue_brunnermunzel_std_confidence_confident_vs_nonconfident',
            _brunnermunzel_pvalue(
                out_df.loc[confident_mask, 'STDConfidence'],
                out_df.loc[non_confident_mask, 'STDConfidence'],
            ),
        ),
    ]

    joint_summary_stats = [
        (
            'joint_mean_confidence_confident',
            pd.to_numeric(joint_confident_values, errors='coerce').mean(),
        ),
        (
            'joint_mean_confidence_nonconfident',
            pd.to_numeric(joint_non_confident_values, errors='coerce').mean(),
        ),
        (
            'joint_median_confidence_confident',
            pd.to_numeric(joint_confident_values, errors='coerce').median(),
        ),
        (
            'joint_median_confidence_nonconfident',
            pd.to_numeric(joint_non_confident_values, errors='coerce').median(),
        ),
        (
            'joint_cohens_d_confidence_confident_vs_nonconfident',
            _fmt_cohens_d_ci(_cohens_d_effect_stats(
                joint_confident_values,
                joint_non_confident_values,
            )),
        ),
        (
            'joint_pvalue_brunnermunzel_confidence_confident_vs_nonconfident',
            _brunnermunzel_pvalue(
                joint_confident_values,
                joint_non_confident_values,
            ),
        ),
    ]

    # Joint threshold is computed from the held-out fold, but the final
    # summarize-only CSV should keep that number next to the other summary rows.
    summary_stats.append(joint_threshold_summary)

    def _fmt3(value):
        val = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]

        if pd.isna(val):
            return np.nan

        val = float(val)

        if val != 0.0 and f'{val:.3f}' in {'0.000', '-0.000'}:
            return f'{val:.3e}'

        return f'{val:.3f}'

    def _fmt_mean_ci(metric_stats):
        if not isinstance(metric_stats, dict):
            return np.nan

        mean = pd.to_numeric(pd.Series([metric_stats.get('mean', np.nan)]), errors='coerce').iloc[0]
        ci_lower = pd.to_numeric(pd.Series([metric_stats.get('ci_lower', np.nan)]), errors='coerce').iloc[0]
        ci_upper = pd.to_numeric(pd.Series([metric_stats.get('ci_upper', np.nan)]), errors='coerce').iloc[0]

        if pd.isna(mean) or pd.isna(ci_lower) or pd.isna(ci_upper):
            return np.nan

        return f'{float(mean):.3f} ({float(ci_lower):.3f}, {float(ci_upper):.3f})'

    def _brier_from_columns(y_true_series, y_pred_series, y_conf_series):
        y_true_num = pd.to_numeric(y_true_series, errors='coerce')
        y_pred_num = pd.to_numeric(y_pred_series, errors='coerce')
        y_conf_num = pd.to_numeric(y_conf_series, errors='coerce')
        valid_mask = y_true_num.notna() & y_pred_num.notna() & y_conf_num.notna()

        if not valid_mask.any():
            return np.nan

        y_true_arr = y_true_num.loc[valid_mask].to_numpy(dtype=float)
        y_pred_arr = y_pred_num.loc[valid_mask].to_numpy(dtype=float)
        y_conf_arr = y_conf_num.loc[valid_mask].to_numpy(dtype=float)

        if np.any(~np.isin(y_true_arr, [0.0, 1.0])):
            return np.nan

        if np.any(~np.isin(y_pred_arr, [0.0, 1.0])):
            return np.nan

        if np.any(~np.isfinite(y_conf_arr)):
            return np.nan

        if np.any((y_conf_arr < 0.0) | (y_conf_arr > 1.0)):
            return np.nan

        y_pos_score = np.where(y_pred_arr.astype(int) == 1, y_conf_arr, 1.0 - y_conf_arr)

        try:
            return float(brier_score_loss(y_true_arr.astype(int), y_pos_score))
        except ValueError:
            return np.nan

    def _balanced_brier_from_columns(y_true_series, y_pred_series, y_conf_series):
        y_true_num = pd.to_numeric(y_true_series, errors='coerce')
        y_pred_num = pd.to_numeric(y_pred_series, errors='coerce')
        y_conf_num = pd.to_numeric(y_conf_series, errors='coerce')
        valid_mask = y_true_num.notna() & y_pred_num.notna() & y_conf_num.notna()

        if not valid_mask.any():
            return np.nan

        y_true_arr = y_true_num.loc[valid_mask].to_numpy(dtype=float)
        y_pred_arr = y_pred_num.loc[valid_mask].to_numpy(dtype=float)
        y_conf_arr = y_conf_num.loc[valid_mask].to_numpy(dtype=float)

        if np.any(~np.isin(y_true_arr, [0.0, 1.0])):
            return np.nan

        if np.any(~np.isin(y_pred_arr, [0.0, 1.0])):
            return np.nan

        if np.any(~np.isfinite(y_conf_arr)):
            return np.nan

        if np.any((y_conf_arr < 0.0) | (y_conf_arr > 1.0)):
            return np.nan

        y_pos_score = np.where(y_pred_arr.astype(int) == 1, y_conf_arr, 1.0 - y_conf_arr)

        y_true_int = y_true_arr.astype(int)

        try:
            wts = compute_sample_weight(class_weight='balanced', y=y_true_int)
            return float(brier_score_loss(y_true_int, y_pos_score, sample_weight=wts))
        except ValueError:
            return np.nan

    def _pr_auc_from_columns(y_true_series, y_pred_series, y_conf_series):
        y_true_num = pd.to_numeric(y_true_series, errors='coerce')
        y_pred_num = pd.to_numeric(y_pred_series, errors='coerce')
        y_conf_num = pd.to_numeric(y_conf_series, errors='coerce')
        valid_mask = y_true_num.notna() & y_pred_num.notna() & y_conf_num.notna()

        if not valid_mask.any():
            return np.nan

        y_true_arr = y_true_num.loc[valid_mask].to_numpy(dtype=float)
        y_pred_arr = y_pred_num.loc[valid_mask].to_numpy(dtype=float)
        y_conf_arr = y_conf_num.loc[valid_mask].to_numpy(dtype=float)

        if np.any(~np.isin(y_true_arr, [0.0, 1.0])):
            return np.nan

        if np.any(~np.isin(y_pred_arr, [0.0, 1.0])):
            return np.nan

        if np.any(~np.isfinite(y_conf_arr)):
            return np.nan

        if np.any((y_conf_arr < 0.0) | (y_conf_arr > 1.0)):
            return np.nan

        y_pos_score = np.where(y_pred_arr.astype(int) == 1, y_conf_arr, 1.0 - y_conf_arr)

        try:
            return float(average_precision_score(y_true_arr.astype(int), y_pos_score))
        except ValueError:
            return np.nan

    def _bootstrap_auc_f1_from_columns(y_true_series, y_pred_series, y_conf_series, threshold):
        y_true_num = pd.to_numeric(y_true_series, errors='coerce')
        y_pred_num = pd.to_numeric(y_pred_series, errors='coerce')
        y_conf_num = pd.to_numeric(y_conf_series, errors='coerce')
        valid_mask = y_true_num.notna() & y_pred_num.notna() & y_conf_num.notna()

        if not valid_mask.any():
            return None

        y_true_arr = y_true_num.loc[valid_mask].to_numpy(dtype=float)
        y_pred_arr = y_pred_num.loc[valid_mask].to_numpy(dtype=float)
        y_conf_arr = y_conf_num.loc[valid_mask].to_numpy(dtype=float)

        if np.any(~np.isin(y_true_arr, [0.0, 1.0])):
            return None

        if np.any(~np.isin(y_pred_arr, [0.0, 1.0])):
            return None

        if np.any(~np.isfinite(y_conf_arr)):
            return None

        if np.any((y_conf_arr < 0.0) | (y_conf_arr > 1.0)):
            return None

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return None

        if threshold < 0.0 or threshold > 1.0:
            return None

        y_pos_score = np.where(y_pred_arr.astype(int) == 1, y_conf_arr, 1.0 - y_conf_arr)
        y_true_int = y_true_arr.astype(int)

        if np.unique(y_true_int).size < 2:
            return None

        return stratified_bootstrap_metrics(
            y_true=y_true_int,
            y_pred=y_pos_score,
            threshold=threshold,
            n_bootstraps=bootstrap_n,
            ci_level=bootstrap_ci,
            seed=bootstrap_seed,
        )

    def _cohens_d_from_bootstrap_stats(stats_joint, stats_physician):
        if not isinstance(stats_joint, dict) or not isinstance(stats_physician, dict):
            return None

        required_keys = ('mean', 'std')
        if any(key not in stats_joint for key in required_keys) or any(key not in stats_physician for key in required_keys):
            return None

        try:
            return cohens_d_from_stats(
                stats_joint['mean'],
                stats_joint['std'],
                bootstrap_n,
                stats_physician['mean'],
                stats_physician['std'],
                bootstrap_n,
                alpha=bootstrap_alpha,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _fmt_summary_value(metric_value):
        if isinstance(metric_value, str):
            return metric_value

        return _fmt3(metric_value)

    def _set_mean_std_from_physician_metric(row, metric_stats_by_physician, metric_key):
        metric_means = []

        for physician_name in model_physicians:
            physician_stats = metric_stats_by_physician.get(physician_name, None)

            if not isinstance(physician_stats, dict):
                continue

            metric_stats = physician_stats.get(metric_key, {})

            if not isinstance(metric_stats, dict):
                continue

            mean_val = pd.to_numeric(
                pd.Series([metric_stats.get('mean', np.nan)]),
                errors='coerce',
            ).iloc[0]

            if pd.notna(mean_val):
                metric_means.append(float(mean_val))

        if not metric_means:
            row['Mean'] = np.nan
            row['STD'] = np.nan
            return

        row['Mean'] = _fmt3(np.mean(metric_means))
        row['STD'] = _fmt3(np.std(metric_means, ddof=1)) if len(metric_means) >= 2 else np.nan

    def _hedges_g_from_physician_metric_means(
        metric_stats_confident_by_physician,
        metric_stats_nonconfident_by_physician,
        metric_key,
    ):
        conf_vals = []
        nonconf_vals = []

        for physician_name in model_physicians:
            stats_conf = metric_stats_confident_by_physician.get(physician_name, None)
            stats_nonconf = metric_stats_nonconfident_by_physician.get(physician_name, None)

            if not isinstance(stats_conf, dict) or not isinstance(stats_nonconf, dict):
                continue

            metric_conf = stats_conf.get(metric_key, {})
            metric_nonconf = stats_nonconf.get(metric_key, {})

            if not isinstance(metric_conf, dict) or not isinstance(metric_nonconf, dict):
                continue

            conf_mean = pd.to_numeric(
                pd.Series([metric_conf.get('mean', np.nan)]),
                errors='coerce',
            ).iloc[0]
            nonconf_mean = pd.to_numeric(
                pd.Series([metric_nonconf.get('mean', np.nan)]),
                errors='coerce',
            ).iloc[0]

            if pd.isna(conf_mean) or pd.isna(nonconf_mean):
                continue

            conf_vals.append(float(conf_mean))
            nonconf_vals.append(float(nonconf_mean))

        if len(conf_vals) < 2 or len(nonconf_vals) < 2:
            return np.nan

        try:
            return float(EffectSizeCalculator(conf_vals, nonconf_vals, paired=True).hedges_g())
        except ValueError:
            return np.nan

    summary_rows = []

    brier_row = {col: np.nan for col in out_df.columns}
    _true_labels_num = pd.to_numeric(out_df['TrueLabel'], errors='coerce').dropna()
    _p_pos = float(_true_labels_num.mean()) if len(_true_labels_num) > 0 else float('nan')
    _dummy_brier = _p_pos * (1.0 - _p_pos)
    brier_row['ScanID'] = f'brier_scores_(dummy={_dummy_brier:.3f})'

    for physician_name in model_physicians:
        conf_col = f'{physician_name}_PredConfidence'

        if conf_col in out_df.columns and physician_name in out_df.columns:
            brier_row[conf_col] = _fmt3(
                _brier_from_columns(out_df['TrueLabel'], out_df[physician_name], out_df[conf_col])
            )

    if 'JointConfidence' in out_df.columns and 'JointLabel' in out_df.columns:
        brier_row['JointConfidence'] = _fmt3(
            _brier_from_columns(out_df['TrueLabel'], out_df['JointLabel'], out_df['JointConfidence'])
        )

    summary_rows.append(brier_row)

    balanced_brier_row = {col: np.nan for col in out_df.columns}
    _dummy_balanced_brier = (_p_pos ** 2) - _p_pos + 0.5
    balanced_brier_row['ScanID'] = f'balanced_brier_scores_(dummy={_dummy_balanced_brier:.3f})'

    for physician_name in model_physicians:
        conf_col = f'{physician_name}_PredConfidence'

        if conf_col in out_df.columns and physician_name in out_df.columns:
            balanced_brier_row[conf_col] = _fmt3(
                _balanced_brier_from_columns(
                    out_df['TrueLabel'],
                    out_df[physician_name],
                    out_df[conf_col],
                )
            )

    if 'JointConfidence' in out_df.columns and 'JointLabel' in out_df.columns:
        balanced_brier_row['JointConfidence'] = _fmt3(
            _balanced_brier_from_columns(
                out_df['TrueLabel'],
                out_df['JointLabel'],
                out_df['JointConfidence'],
            )
        )

    summary_rows.append(balanced_brier_row)

    joint_bootstrap_stats = None

    if 'JointConfidence' in out_df.columns and 'JointLabel' in out_df.columns:
        joint_bootstrap_stats = _bootstrap_auc_f1_from_columns(
            out_df['TrueLabel'],
            out_df['JointLabel'],
            out_df['JointConfidence'],
            threshold=joint_threshold,
        )

    joint_bootstrap_stats_confident = None
    joint_bootstrap_stats_nonconfident = None

    if 'JointConfidence' in out_df.columns and 'JointLabel' in out_df.columns:
        joint_bootstrap_stats_confident = _bootstrap_auc_f1_from_columns(
            out_df.loc[confident_mask, 'TrueLabel'],
            out_df.loc[confident_mask, 'JointLabel'],
            out_df.loc[confident_mask, 'JointConfidence'],
            threshold=joint_threshold,
        )
        joint_bootstrap_stats_nonconfident = _bootstrap_auc_f1_from_columns(
            out_df.loc[non_confident_mask, 'TrueLabel'],
            out_df.loc[non_confident_mask, 'JointLabel'],
            out_df.loc[non_confident_mask, 'JointConfidence'],
            threshold=joint_threshold,
        )

    pr_auc_row = {col: np.nan for col in out_df.columns}
    pr_auc_row['ScanID'] = 'PR_AUC'
    physician_bootstrap_by_name = {}
    physician_bootstrap_confident_by_name = {}
    physician_bootstrap_nonconfident_by_name = {}

    for physician_name in model_physicians:
        conf_col = f'{physician_name}_PredConfidence'

        if conf_col in out_df.columns and physician_name in out_df.columns:
            physician_threshold = decision_threshold_by_physician.get(physician_name, 0.5)
            physician_bootstrap_stats = _bootstrap_auc_f1_from_columns(
                out_df['TrueLabel'],
                out_df[physician_name],
                out_df[conf_col],
                threshold=physician_threshold,
            )
            physician_bootstrap_conf = _bootstrap_auc_f1_from_columns(
                out_df.loc[confident_mask, 'TrueLabel'],
                out_df.loc[confident_mask, physician_name],
                out_df.loc[confident_mask, conf_col],
                threshold=physician_threshold,
            )
            physician_bootstrap_nonconf = _bootstrap_auc_f1_from_columns(
                out_df.loc[non_confident_mask, 'TrueLabel'],
                out_df.loc[non_confident_mask, physician_name],
                out_df.loc[non_confident_mask, conf_col],
                threshold=physician_threshold,
            )
            physician_bootstrap_by_name[physician_name] = physician_bootstrap_stats
            physician_bootstrap_confident_by_name[physician_name] = physician_bootstrap_conf
            physician_bootstrap_nonconfident_by_name[physician_name] = physician_bootstrap_nonconf

            if physician_bootstrap_stats is not None:
                pr_auc_row[conf_col] = _fmt_mean_ci(physician_bootstrap_stats.get('pr_auc', {}))

    if 'JointConfidence' in out_df.columns and joint_bootstrap_stats is not None:
        pr_auc_row['JointConfidence'] = _fmt_mean_ci(joint_bootstrap_stats.get('pr_auc', {}))

    _set_mean_std_from_physician_metric(pr_auc_row, physician_bootstrap_by_name, 'pr_auc')

    summary_rows.append(pr_auc_row)

    pr_auc_confident_row = {col: np.nan for col in out_df.columns}
    pr_auc_confident_row['ScanID'] = 'PR_AUC_confident'

    pr_auc_nonconfident_row = {col: np.nan for col in out_df.columns}
    pr_auc_nonconfident_row['ScanID'] = 'PR_AUC_nonconfident'

    for physician_name in model_physicians:
        conf_col = f'{physician_name}_PredConfidence'
        physician_bootstrap_conf = physician_bootstrap_confident_by_name.get(physician_name, None)
        physician_bootstrap_nonconf = physician_bootstrap_nonconfident_by_name.get(physician_name, None)

        if conf_col in out_df.columns and physician_name in out_df.columns:
            if physician_bootstrap_conf is not None:
                pr_auc_confident_row[conf_col] = _fmt_mean_ci(physician_bootstrap_conf.get('pr_auc', {}))

            if physician_bootstrap_nonconf is not None:
                pr_auc_nonconfident_row[conf_col] = _fmt_mean_ci(physician_bootstrap_nonconf.get('pr_auc', {}))

    if 'JointConfidence' in out_df.columns and joint_bootstrap_stats_confident is not None:
        pr_auc_confident_row['JointConfidence'] = _fmt_mean_ci(
            joint_bootstrap_stats_confident.get('pr_auc', {})
        )

    if 'JointConfidence' in out_df.columns and joint_bootstrap_stats_nonconfident is not None:
        pr_auc_nonconfident_row['JointConfidence'] = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('pr_auc', {})
        )

    _set_mean_std_from_physician_metric(pr_auc_confident_row, physician_bootstrap_confident_by_name, 'pr_auc')
    _set_mean_std_from_physician_metric(pr_auc_nonconfident_row, physician_bootstrap_nonconfident_by_name, 'pr_auc')

    summary_rows.append(pr_auc_confident_row)
    summary_rows.append(pr_auc_nonconfident_row)

    pr_auc_hedges_g_row = {col: np.nan for col in out_df.columns}
    pr_auc_hedges_g_row['ScanID'] = 'PR_AUC_hedges_g_conf_vs_nonconf'
    pr_auc_hedges_g_row[first_physician_conf_col] = _fmt3(
        _hedges_g_from_physician_metric_means(
            physician_bootstrap_confident_by_name,
            physician_bootstrap_nonconfident_by_name,
            'pr_auc',
        )
    )
    summary_rows.append(pr_auc_hedges_g_row)

    def _f1_from_columns(y_true, y_pred):
        y_true_num = pd.to_numeric(y_true, errors='coerce')
        y_pred_num = pd.to_numeric(y_pred, errors='coerce')
        valid_mask = y_true_num.notna() & y_pred_num.notna()

        if not valid_mask.any():
            return np.nan

        y_true_arr = y_true_num.loc[valid_mask].to_numpy(dtype=int)
        y_pred_arr = y_pred_num.loc[valid_mask].to_numpy(dtype=int)

        if np.any(~np.isin(y_true_arr, [0, 1])) or np.any(~np.isin(y_pred_arr, [0, 1])):
            return np.nan

        try:
            return float(f1_score(y_true_arr, y_pred_arr))
        except ValueError:
            return np.nan

    f1_row = {col: np.nan for col in out_df.columns}
    f1_row['ScanID'] = 'F1_score'

    for physician_name in model_physicians:
        pred_col = physician_name
        conf_col = f'{physician_name}_PredConfidence'

        if pred_col in out_df.columns and conf_col in out_df.columns:
            physician_bootstrap_stats = physician_bootstrap_by_name.get(physician_name, None)

            if physician_bootstrap_stats is not None:
                f1_row[conf_col] = _fmt_mean_ci(physician_bootstrap_stats.get('f1_score', {}))

    if 'JointLabel' in out_df.columns and joint_bootstrap_stats is not None:
        f1_row['JointConfidence'] = _fmt_mean_ci(joint_bootstrap_stats.get('f1_score', {}))

    _set_mean_std_from_physician_metric(f1_row, physician_bootstrap_by_name, 'f1_score')

    summary_rows.append(f1_row)

    f1_confident_row = {col: np.nan for col in out_df.columns}
    f1_confident_row['ScanID'] = 'F1_confident'

    f1_nonconfident_row = {col: np.nan for col in out_df.columns}
    f1_nonconfident_row['ScanID'] = 'F1_nonconfident'

    for physician_name in model_physicians:
        pred_col = physician_name
        conf_col = f'{physician_name}_PredConfidence'
        physician_bootstrap_conf = physician_bootstrap_confident_by_name.get(physician_name, None)
        physician_bootstrap_nonconf = physician_bootstrap_nonconfident_by_name.get(physician_name, None)

        if pred_col in out_df.columns and conf_col in out_df.columns:
            if physician_bootstrap_conf is not None:
                f1_confident_row[conf_col] = _fmt_mean_ci(physician_bootstrap_conf.get('f1_score', {}))

            if physician_bootstrap_nonconf is not None:
                f1_nonconfident_row[conf_col] = _fmt_mean_ci(physician_bootstrap_nonconf.get('f1_score', {}))

    if 'JointLabel' in out_df.columns and joint_bootstrap_stats_confident is not None:
        f1_confident_row['JointConfidence'] = _fmt_mean_ci(
            joint_bootstrap_stats_confident.get('f1_score', {})
        )

    if 'JointLabel' in out_df.columns and joint_bootstrap_stats_nonconfident is not None:
        f1_nonconfident_row['JointConfidence'] = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('f1_score', {})
        )

    _set_mean_std_from_physician_metric(f1_confident_row, physician_bootstrap_confident_by_name, 'f1_score')
    _set_mean_std_from_physician_metric(f1_nonconfident_row, physician_bootstrap_nonconfident_by_name, 'f1_score')

    summary_rows.append(f1_confident_row)
    summary_rows.append(f1_nonconfident_row)

    f1_hedges_g_row = {col: np.nan for col in out_df.columns}
    f1_hedges_g_row['ScanID'] = 'F1_hedges_g_conf_vs_nonconf'
    f1_hedges_g_row[first_physician_conf_col] = _fmt3(
        _hedges_g_from_physician_metric_means(
            physician_bootstrap_confident_by_name,
            physician_bootstrap_nonconfident_by_name,
            'f1_score',
        )
    )
    summary_rows.append(f1_hedges_g_row)

    joint_pr_auc_row = {col: np.nan for col in out_df.columns}
    joint_pr_auc_row['ScanID'] = 'joint_PR_AUC'

    if joint_bootstrap_stats is not None:
        joint_pr_auc_row[first_physician_conf_col] = _fmt_mean_ci(joint_bootstrap_stats.get('pr_auc', {}))

    summary_rows.append(joint_pr_auc_row)

    joint_pr_auc_confident_row = {col: np.nan for col in out_df.columns}
    joint_pr_auc_confident_row['ScanID'] = 'joint_PR_AUC_confident'

    if joint_bootstrap_stats_confident is not None:
        joint_pr_auc_confident_row[first_physician_conf_col] = _fmt_mean_ci(
            joint_bootstrap_stats_confident.get('pr_auc', {})
        )

    summary_rows.append(joint_pr_auc_confident_row)

    joint_pr_auc_nonconfident_row = {col: np.nan for col in out_df.columns}
    joint_pr_auc_nonconfident_row['ScanID'] = 'joint_PR_AUC_nonconfident'

    if joint_bootstrap_stats_nonconfident is not None:
        joint_pr_auc_nonconfident_row[first_physician_conf_col] = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('pr_auc', {})
        )

    summary_rows.append(joint_pr_auc_nonconfident_row)

    joint_f1_row = {col: np.nan for col in out_df.columns}
    joint_f1_row['ScanID'] = 'joint_F1'

    if joint_bootstrap_stats is not None:
        joint_f1_row[first_physician_conf_col] = _fmt_mean_ci(joint_bootstrap_stats.get('f1_score', {}))

    summary_rows.append(joint_f1_row)

    joint_f1_confident_row = {col: np.nan for col in out_df.columns}
    joint_f1_confident_row['ScanID'] = 'joint_F1_confident'

    if joint_bootstrap_stats_confident is not None:
        joint_f1_confident_row[first_physician_conf_col] = _fmt_mean_ci(
            joint_bootstrap_stats_confident.get('f1_score', {})
        )

    summary_rows.append(joint_f1_confident_row)

    joint_f1_nonconfident_row = {col: np.nan for col in out_df.columns}
    joint_f1_nonconfident_row['ScanID'] = 'joint_F1_nonconfident'

    if joint_bootstrap_stats_nonconfident is not None:
        joint_f1_nonconfident_row[first_physician_conf_col] = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('f1_score', {})
        )

    summary_rows.append(joint_f1_nonconfident_row)

    for metric_name, metric_value in joint_summary_stats:
        row = {col: np.nan for col in out_df.columns}
        row['ScanID'] = metric_name
        row[first_physician_conf_col] = _fmt_summary_value(metric_value)
        summary_rows.append(row)

    auc_cohens_d_row = {col: np.nan for col in out_df.columns}
    auc_cohens_d_row['ScanID'] = 'AUC_cohens_d'

    auc_cohens_d_confident_row = {col: np.nan for col in out_df.columns}
    auc_cohens_d_confident_row['ScanID'] = 'AUC_cohens_d_confident'

    auc_cohens_d_nonconfident_row = {col: np.nan for col in out_df.columns}
    auc_cohens_d_nonconfident_row['ScanID'] = 'AUC_cohens_d_nonconfident'

    f1_cohens_d_row = {col: np.nan for col in out_df.columns}
    f1_cohens_d_row['ScanID'] = 'F1_cohens_d'

    f1_cohens_d_confident_row = {col: np.nan for col in out_df.columns}
    f1_cohens_d_confident_row['ScanID'] = 'F1_cohens_d_confident'

    f1_cohens_d_nonconfident_row = {col: np.nan for col in out_df.columns}
    f1_cohens_d_nonconfident_row['ScanID'] = 'F1_cohens_d_nonconfident'

    if joint_bootstrap_stats is not None:
        for physician_name in model_physicians:
            conf_col = f'{physician_name}_PredConfidence'

            if conf_col not in out_df.columns:
                continue

            physician_bootstrap_stats = physician_bootstrap_by_name.get(physician_name, None)
            if physician_bootstrap_stats is None:
                continue

            auc_effect_stats = _cohens_d_from_bootstrap_stats(
                joint_bootstrap_stats.get('pr_auc', {}),
                physician_bootstrap_stats.get('pr_auc', {}),
            )
            f1_effect_stats = _cohens_d_from_bootstrap_stats(
                joint_bootstrap_stats.get('f1_score', {}),
                physician_bootstrap_stats.get('f1_score', {}),
            )

            physician_bootstrap_stats_confident = physician_bootstrap_confident_by_name.get(
                physician_name,
                None,
            )
            physician_bootstrap_stats_nonconfident = physician_bootstrap_nonconfident_by_name.get(
                physician_name,
                None,
            )

            auc_effect_stats_confident = _cohens_d_from_bootstrap_stats(
                (joint_bootstrap_stats_confident or {}).get('pr_auc', {}),
                (physician_bootstrap_stats_confident or {}).get('pr_auc', {}),
            )
            auc_effect_stats_nonconfident = _cohens_d_from_bootstrap_stats(
                (joint_bootstrap_stats_nonconfident or {}).get('pr_auc', {}),
                (physician_bootstrap_stats_nonconfident or {}).get('pr_auc', {}),
            )
            f1_effect_stats_confident = _cohens_d_from_bootstrap_stats(
                (joint_bootstrap_stats_confident or {}).get('f1_score', {}),
                (physician_bootstrap_stats_confident or {}).get('f1_score', {}),
            )
            f1_effect_stats_nonconfident = _cohens_d_from_bootstrap_stats(
                (joint_bootstrap_stats_nonconfident or {}).get('f1_score', {}),
                (physician_bootstrap_stats_nonconfident or {}).get('f1_score', {}),
            )

            if auc_effect_stats is not None:
                auc_cohens_d_row[conf_col] = _fmt_cohens_d_ci(auc_effect_stats)

            if auc_effect_stats_confident is not None:
                auc_cohens_d_confident_row[conf_col] = _fmt_cohens_d_ci(auc_effect_stats_confident)

            if auc_effect_stats_nonconfident is not None:
                auc_cohens_d_nonconfident_row[conf_col] = _fmt_cohens_d_ci(auc_effect_stats_nonconfident)

            if f1_effect_stats is not None:
                f1_cohens_d_row[conf_col] = _fmt_cohens_d_ci(f1_effect_stats)

            if f1_effect_stats_confident is not None:
                f1_cohens_d_confident_row[conf_col] = _fmt_cohens_d_ci(f1_effect_stats_confident)

            if f1_effect_stats_nonconfident is not None:
                f1_cohens_d_nonconfident_row[conf_col] = _fmt_cohens_d_ci(f1_effect_stats_nonconfident)

    summary_rows.append(auc_cohens_d_row)
    summary_rows.append(auc_cohens_d_confident_row)
    summary_rows.append(auc_cohens_d_nonconfident_row)
    summary_rows.append(f1_cohens_d_row)
    summary_rows.append(f1_cohens_d_confident_row)
    summary_rows.append(f1_cohens_d_nonconfident_row)

    for metric_name, metric_value in summary_stats:
        row = {col: np.nan for col in out_df.columns}
        row['ScanID'] = metric_name
        row[first_physician_conf_col] = _fmt_summary_value(metric_value)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=out_df.columns)
    out_df = pd.concat([out_df, summary_df], axis=0, ignore_index=True)

    threshold_methods = sorted(set(threshold_selection_by_csv.values()))

    if len(threshold_methods) != 1:
        raise ValueError(
            'Inconsistent ThresholdSelection across predict CSV inputs. '
            f'Found methods={threshold_methods}. '
            f'Per-file map={threshold_selection_by_csv}'
        )

    threshold_selection = threshold_methods[0]
    print(f'Resolved ThresholdSelection from input CSVs: {threshold_selection}')

    out_csv = _build_predict_summarize_csv_path(
        ss,
        summarize_cfg,
        model_cfg,
        model_physicians,
        target_physicians,
        threshold_selection,
        calibration_method=_resolve_predict_calibration_method(predict_cfg),
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f'Loaded {len(model_physicians) * 2} predict CSV files for summarize-only mode.')
    print(f'Summarized per-scan table saved to: {out_csv}')
    print('Summary stats appended at bottom of CSV:')

    pr_auc_summary_text = np.nan
    pr_auc_confident_summary_text = np.nan
    pr_auc_nonconfident_summary_text = np.nan
    f1_summary_text = np.nan
    f1_confident_summary_text = np.nan
    f1_nonconfident_summary_text = np.nan
    joint_pr_auc_confident_summary_text = np.nan
    joint_pr_auc_nonconfident_summary_text = np.nan
    joint_f1_summary_text = np.nan
    joint_f1_confident_summary_text = np.nan
    joint_f1_nonconfident_summary_text = np.nan
    pr_auc_hedges_g_summary_text = _fmt3(
        _hedges_g_from_physician_metric_means(
            physician_bootstrap_confident_by_name,
            physician_bootstrap_nonconfident_by_name,
            'pr_auc',
        )
    )
    f1_hedges_g_summary_text = _fmt3(
        _hedges_g_from_physician_metric_means(
            physician_bootstrap_confident_by_name,
            physician_bootstrap_nonconfident_by_name,
            'f1_score',
        )
    )

    if joint_bootstrap_stats is not None:
        pr_auc_summary_text = _fmt_mean_ci(joint_bootstrap_stats.get('pr_auc', {}))
        f1_summary_text = _fmt_mean_ci(joint_bootstrap_stats.get('f1_score', {}))
        joint_f1_summary_text = _fmt_mean_ci(joint_bootstrap_stats.get('f1_score', {}))

    if joint_bootstrap_stats_confident is not None:
        pr_auc_confident_summary_text = _fmt_mean_ci(joint_bootstrap_stats_confident.get('pr_auc', {}))
        f1_confident_summary_text = _fmt_mean_ci(joint_bootstrap_stats_confident.get('f1_score', {}))
        joint_pr_auc_confident_summary_text = _fmt_mean_ci(
            joint_bootstrap_stats_confident.get('pr_auc', {})
        )
        joint_f1_confident_summary_text = _fmt_mean_ci(joint_bootstrap_stats_confident.get('f1_score', {}))

    if joint_bootstrap_stats_nonconfident is not None:
        pr_auc_nonconfident_summary_text = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('pr_auc', {})
        )
        f1_nonconfident_summary_text = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('f1_score', {})
        )
        joint_pr_auc_nonconfident_summary_text = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('pr_auc', {})
        )
        joint_f1_nonconfident_summary_text = _fmt_mean_ci(
            joint_bootstrap_stats_nonconfident.get('f1_score', {})
        )

    print(f'  PR_AUC: {pr_auc_summary_text}')
    print(f'  PR_AUC_confident: {pr_auc_confident_summary_text}')
    print(f'  PR_AUC_nonconfident: {pr_auc_nonconfident_summary_text}')
    print(f'  PR_AUC_hedges_g_conf_vs_nonconf: {pr_auc_hedges_g_summary_text}')
    print(f'  F1_score: {f1_summary_text}')
    print(f'  F1_confident: {f1_confident_summary_text}')
    print(f'  F1_nonconfident: {f1_nonconfident_summary_text}')
    print(f'  F1_hedges_g_conf_vs_nonconf: {f1_hedges_g_summary_text}')
    print(f'  joint_PR_AUC: {pr_auc_summary_text}')
    print(f'  joint_PR_AUC_confident: {joint_pr_auc_confident_summary_text}')
    print(f'  joint_PR_AUC_nonconfident: {joint_pr_auc_nonconfident_summary_text}')
    print(f'  joint_F1: {joint_f1_summary_text}')
    print(f'  joint_F1_confident: {joint_f1_confident_summary_text}')
    print(f'  joint_F1_nonconfident: {joint_f1_nonconfident_summary_text}')

    for metric_name, metric_value in joint_summary_stats:
        fmt_value = _fmt_summary_value(metric_value)
        print(f'  {metric_name}: {fmt_value}')

    auc_cohens_d_text = []
    auc_cohens_d_confident_text = []
    auc_cohens_d_nonconfident_text = []
    f1_cohens_d_text = []
    f1_cohens_d_confident_text = []
    f1_cohens_d_nonconfident_text = []

    for physician_name in model_physicians:
        conf_col = f'{physician_name}_PredConfidence'

        if pd.notna(auc_cohens_d_row.get(conf_col, np.nan)):
            auc_cohens_d_text.append(f'{conf_col}={auc_cohens_d_row[conf_col]}')

        if pd.notna(auc_cohens_d_confident_row.get(conf_col, np.nan)):
            auc_cohens_d_confident_text.append(
                f'{conf_col}={auc_cohens_d_confident_row[conf_col]}'
            )

        if pd.notna(auc_cohens_d_nonconfident_row.get(conf_col, np.nan)):
            auc_cohens_d_nonconfident_text.append(
                f'{conf_col}={auc_cohens_d_nonconfident_row[conf_col]}'
            )

        if pd.notna(f1_cohens_d_row.get(conf_col, np.nan)):
            f1_cohens_d_text.append(f'{conf_col}={f1_cohens_d_row[conf_col]}')

        if pd.notna(f1_cohens_d_confident_row.get(conf_col, np.nan)):
            f1_cohens_d_confident_text.append(
                f'{conf_col}={f1_cohens_d_confident_row[conf_col]}'
            )

        if pd.notna(f1_cohens_d_nonconfident_row.get(conf_col, np.nan)):
            f1_cohens_d_nonconfident_text.append(
                f'{conf_col}={f1_cohens_d_nonconfident_row[conf_col]}'
            )

    print(f'  AUC_cohens_d: {", ".join(auc_cohens_d_text) if auc_cohens_d_text else "nan"}')
    print(
        '  AUC_cohens_d_confident: '
        f'{", ".join(auc_cohens_d_confident_text) if auc_cohens_d_confident_text else "nan"}'
    )
    print(
        '  AUC_cohens_d_nonconfident: '
        f'{", ".join(auc_cohens_d_nonconfident_text) if auc_cohens_d_nonconfident_text else "nan"}'
    )
    print(f'  F1_cohens_d: {", ".join(f1_cohens_d_text) if f1_cohens_d_text else "nan"}')
    print(
        '  F1_cohens_d_confident: '
        f'{", ".join(f1_cohens_d_confident_text) if f1_cohens_d_confident_text else "nan"}'
    )
    print(
        '  F1_cohens_d_nonconfident: '
        f'{", ".join(f1_cohens_d_nonconfident_text) if f1_cohens_d_nonconfident_text else "nan"}'
    )

    for metric_name, metric_value in summary_stats:
        fmt_value = _fmt_summary_value(metric_value)
        print(f'  {metric_name}: {fmt_value}')


def _build_summarize_model_cfg(ss, predict_cfg, summarize_cfg):
    """Build synthetic model settings used to reconstruct predict CSV names."""
    predict_model_cfg = predict_cfg.get('model', {}) or {}

    hospital = summarize_cfg.get('hospital', None)

    if not hospital:
        raise ValueError('predict.summarize.hospital must be specified for predict.summarize_only mode.')

    return {
        'hospital': hospital,
        'target_label': summarize_cfg.get('target_label', predict_model_cfg.get('target_label', ss.args.get('target_label'))),
        'dim_reduction': summarize_cfg.get('dim_reduction', predict_model_cfg.get('dim_reduction', ss.args.get('dim_reduction'))),
        'use_moments_only': summarize_cfg.get(
            'use_moments_only',
            predict_model_cfg.get('use_moments_only', ss.args.get('use_moments_only', True)),
        ),
        'standardize_features': summarize_cfg.get(
            'standardize_features',
            predict_model_cfg.get('standardize_features', ss.args.get('standardize_features', False)),
        ),
        'ignore_confidence': summarize_cfg.get(
            'ignore_confidence',
            predict_model_cfg.get('ignore_confidence', ss.args.get('ignore_confidence', False)),
        ),
        'use_consensus_cv': summarize_cfg.get('use_consensus_cv', predict_model_cfg.get('use_consensus_cv', None)),
        'threshold_selection': summarize_cfg.get('threshold_selection', predict_model_cfg.get('threshold_selection', None)),
    }


def _build_predict_summarize_csv_path(
    ss,
    summarize_cfg,
    model_cfg,
    model_physicians,
    target_physicians,
    threshold_selection,
    calibration_method=None,
):
    """Build dedicated CSV output path for summarize-only predict mode."""
    model_token = physician_token(model_physicians)
    target_token = physician_token(target_physicians)
    dim_token = str(summarize_cfg.get('dim_reduction', model_cfg.get('dim_reduction', 'none')))
    label_token = str(model_cfg.get('target_label', 'label'))
    ignore_conf = bool(model_cfg.get('ignore_confidence', False))
    nparms = 5 if bool(model_cfg.get('use_moments_only', True)) else 9
    threshold_suffix = _threshold_selection_file_suffix(threshold_selection)
    calibration_suffix = _predict_calibration_file_suffix(calibration_method)

    fname = (
        f'predict_summary_{model_token}2{target_token}_{label_token}_{dim_token}_'
        f'nparms{nparms}_std{bool(model_cfg.get("standardize_features", False))}_'
        f'ignoreConf{ignore_conf}{calibration_suffix}{threshold_suffix}.csv'
    )
    return _predict_results_root(ss) / fname


def _threshold_selection_file_suffix(threshold_selection):
    """Return summary filename suffix encoding predict threshold strategy."""
    if threshold_selection is None:
        return '_T05'

    method = str(threshold_selection).strip().lower()

    if not method or method == 'default':
        return '_T05'

    if method == 'f1':
        return '_f1'

    safe = re.sub(r'[^a-z0-9]+', '', method)

    if not safe:
        return '_T05'

    return f'_{safe}'


def _extract_csv_threshold_selection(df, csv_path):
    """Return normalized ThresholdSelection from one predict CSV and validate consistency."""
    if 'ThresholdSelection' not in df.columns:
        raise ValueError(
            f'CSV file {csv_path} is missing required column: ThresholdSelection. '
            'Summarize-only mode requires per-file threshold metadata.'
        )

    raw_vals = [str(v).strip().lower() for v in df['ThresholdSelection'].tolist() if pd.notna(v)]

    if not raw_vals:
        raise ValueError(
            f'CSV file {csv_path} has empty ThresholdSelection values. '
            'Summarize-only mode requires a valid threshold method in each input CSV.'
        )

    normalized_vals = ['default' if (not v or v == 'none') else v for v in raw_vals]
    uniq = sorted(set(normalized_vals))

    if len(uniq) != 1:
        raise ValueError(
            f'CSV file {csv_path} has inconsistent ThresholdSelection values: {uniq}'
        )

    return uniq[0]


def _extract_csv_decision_threshold(df, csv_path):
    """Return decision threshold from one predict CSV and validate consistency."""
    if 'DecisionThreshold' not in df.columns:
        raise ValueError(
            f'CSV file {csv_path} is missing required column: DecisionThreshold. '
            'Summarize-only mode requires per-file threshold values for bootstrap F1.'
        )

    raw_vals = pd.to_numeric(df['DecisionThreshold'], errors='coerce')
    raw_vals = raw_vals[np.isfinite(raw_vals)]

    if raw_vals.empty:
        raise ValueError(
            f'CSV file {csv_path} has empty DecisionThreshold values. '
            'Summarize-only mode requires a valid decision threshold in each input CSV.'
        )

    uniq = np.unique(raw_vals.to_numpy(dtype=float))

    if uniq.size != 1:
        raise ValueError(
            f'CSV file {csv_path} has inconsistent DecisionThreshold values: {uniq.tolist()}'
        )

    threshold = float(uniq[0])

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f'CSV file {csv_path} has out-of-range DecisionThreshold value: {threshold}'
        )

    return threshold


def _predict_results_root(ss):
    """Return directory where predict-related CSV files are stored/read from."""
    predict_cfg = ss.args.get('predict', {}) or {}
    subfolder = predict_cfg.get('results_subfolder', None)

    if subfolder is None:
        return ss.out_root

    subfolder_text = str(subfolder).strip()

    if not subfolder_text:
        return ss.out_root

    return ss.out_root / subfolder_text


def _normalize_physician_list(value, field_name):
    """Normalize physician configuration into a clean list of unique names."""
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f'{field_name} must be a list of physician names, a string, or null.')

    names = []

    for item in value:
        if item is None:
            continue

        text = str(item).strip()

        if not text:
            continue

        if text not in names:
            names.append(text)

    return names


def _load_predict_results_csv(csv_path):
    """Load one predict output CSV and validate required columns/uniqueness."""
    df = pd.read_csv(csv_path)

    required = ('ScanID', 'TrueLabel', 'PredLabel')
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f'CSV file {csv_path} is missing required columns: {missing}')

    if df['ScanID'].duplicated().any():
        dup_ids = df.loc[df['ScanID'].duplicated(), 'ScanID'].tolist()
        raise ValueError(f'CSV file {csv_path} contains duplicate ScanID values: {dup_ids[:10]}')

    return df


def _merge_true_labels(true_label_by_scan, df, csv_path):
    """Merge TrueLabel values into map and enforce cross-file consistency."""
    for scan_id, true_label in zip(df['ScanID'], df['TrueLabel']):
        if scan_id in true_label_by_scan and true_label_by_scan[scan_id] != true_label:
            raise ValueError(
                f'Inconsistent TrueLabel for scan_id={scan_id}: '
                f'{true_label_by_scan[scan_id]} vs {true_label} in {csv_path}'
            )

        true_label_by_scan[scan_id] = true_label


def _build_physician_pair(model_physician, target_physician, input_ignore_confidence):
    model_token = physician_token(model_physician)
    target_token = physician_token(target_physician)
    # Suffix encodes confidence-level selection for predict.input_records.
    # False -> Conf (confidence-preserved labels), True -> All (confidence-collapsed labels).
    conf_level = 'All' if bool(input_ignore_confidence) else 'Conf'
    return f'{model_token}2{target_token}{conf_level}'


def _resolve_model_pickle_path(ss, step_cfg, step_name='predict'):
    """
    Resolve the fully-qualified path to the trained model pickle.

    Top-level ``pickle`` has highest priority and is treated as a path
    relative to ``out_root``.

    If top-level ``pickle`` is null, the name is derived from ``<step_name>.model``
    parameters using the same naming convention as the ``xgboost`` step, by temporarily
    overriding the relevant ``ss.args`` keys so that ``ss.cls_pkl_pname``
    generates the correct name.

    Optionally, ``predict.model.physician`` may specify a non-empty list of
    physician names. In that case a physician-specific suffix token is appended
    to the auto-derived model filename before ``.pkl``.

    Args:
        ss(obj): reference to this app object
        step_cfg(dict): step-specific section of the input JSON
        step_name(str): top-level step key for error text (for example,
            ``predict`` or ``calibrate``)

    Returns:
        pkl_pname(Path): fully-resolved path to the pickle file

    """
    explicit_top_level = ss.resolve_top_level_pickle_override()

    if explicit_top_level is not None:
        return explicit_top_level

    model_cfg = step_cfg.get('model', {}) or {}
    model_physician = model_cfg.get('physician', None)
    pkl_pname = _build_model_pickle_path_from_model_cfg(ss, model_cfg, step_name=step_name)

    # Keep current behavior for null/empty physician settings.
    # For non-empty settings append physician token, matching training-side tag style.
    physician_suffix = physician_token(model_physician)

    if physician_suffix != 'All':
        pkl_pname = pkl_pname.with_name(f'{pkl_pname.stem}_{physician_suffix}{pkl_pname.suffix}')

    return pkl_pname


def _build_model_pickle_path_from_model_cfg(ss, model_cfg, step_name='predict'):
    """Build model pickle path from model-like settings without top-level override."""
    hospital = model_cfg.get('hospital')
    label = model_cfg.get('target_label')
    use_moments_only = bool(model_cfg.get('use_moments_only', True))
    standardize = bool(model_cfg.get('standardize_features', False))
    ignore_confidence = bool(model_cfg.get('ignore_confidence', False))

    if not hospital:
        raise ValueError(
            f'"{step_name}.model.hospital" must be specified when top-level "pickle" is null'
        )

    if not label:
        raise ValueError(
            f'"{step_name}.model.target_label" must be specified when top-level "pickle" is null'
        )

    nparms = 5 if use_moments_only else 9
    hlist = ss.hlist(hospital)

    # Temporarily override reduction-related args so that the internal
    # _reducer_pickle_tag() helper inside ss.cls_pkl_pname returns the right
    # tag for the model we're looking for.
    tag_keys = ('dim_reduction', 'use_consensus_cv')
    saved = {k: copy.deepcopy(ss.args.get(k)) for k in tag_keys}
    dim_method = str(model_cfg.get('dim_reduction', '')).lower()

    if 'dim_reduction' in model_cfg:
        ss.args['dim_reduction'] = model_cfg['dim_reduction']

    if 'use_consensus_cv' in model_cfg and model_cfg.get('use_consensus_cv') is not None:
        ss.args['use_consensus_cv'] = model_cfg['use_consensus_cv']
    else:
        ss.args['use_consensus_cv'] = dim_method in ('ccv', 'ccv-fcls')

    try:
        return ss.cls_pkl_pname(hlist, label, nparms, standardize, ignore_confidence)
    finally:
        for k, v in saved.items():
            ss.args[k] = v


def _configure_predict_model_threads(model, model_cfg):
    """
    Optionally set inference thread count for loaded XGBoost model.

    Configuration key:
        predict.model.n_jobs
    """
    if not isinstance(model_cfg, dict):
        return

    n_jobs = model_cfg.get('n_jobs', None)

    if n_jobs is None:
        return

    try:
        n_jobs = int(n_jobs)
    except (TypeError, ValueError) as exc:
        raise ValueError('predict.model.n_jobs must be an integer when provided.') from exc

    model.set_params(n_jobs=n_jobs)

    try:
        model.get_booster().set_param({'nthread': n_jobs})
    except Exception:
        # Keep prediction functional even if backend rejects runtime thread override.
        pass

    print(f'Predict-time XGBoost threads override applied: n_jobs={n_jobs}')


def _predict_labels_with_thresholding(
    y_proba,
    y_true,
    pkl_pname,
    threshold_selection=None,
    model_y_true=None,
    model_y_score=None,
):
    """
    Predict binary labels from class probabilities using selected threshold strategy.

    Returns:
        tuple: (y_pred, confidence, threshold, method)
    """
    y_proba = np.asarray(y_proba)

    if y_proba.ndim != 2 or y_proba.shape[1] != 2:
        raise ValueError(
            'Threshold-based prediction in cls_predict requires binary probabilities '
            'with shape (n_samples, 2).'
        )

    pos_proba = y_proba[:, 1]

    if threshold_selection is None:
        method = 'default'
        threshold = 0.5
    else:
        if not isinstance(threshold_selection, str):
            raise ValueError('predict.model.threshold_selection must be null or a string.')

        method = threshold_selection.strip().lower()

        if not method:
            raise ValueError('predict.model.threshold_selection cannot be an empty string.')

        threshold = _get_or_optimize_threshold_for_model(
            pkl_pname=pkl_pname,
            method=method,
            input_y_true=y_true,
            input_y_score=pos_proba,
            model_y_true=model_y_true,
            model_y_score=model_y_score,
        )

    y_pred = (pos_proba > threshold).astype(int)
    confidence = _predicted_label_probability_confidence(pos_proba, y_pred)
    # retired: threshold-normalized confidence is no longer used.
    # confidence = _normalized_threshold_confidence(pos_proba, threshold)
    return y_pred, confidence, float(threshold), method


def _get_or_optimize_threshold_for_model(
    pkl_pname,
    method,
    input_y_true,
    input_y_score,
    model_y_true=None,
    model_y_score=None,
):
    """
    Return threshold for model+method, using in-memory static cache if available.
    """
    cache = getattr(_get_or_optimize_threshold_for_model, '_threshold_cache', None)

    if cache is None:
        cache = {}
        _get_or_optimize_threshold_for_model._threshold_cache = cache

    cache_key = (str(Path(pkl_pname).resolve()), method)

    if cache_key in cache:
        return float(cache[cache_key])

    y_true_opt = None
    y_score_opt = None

    if model_y_true is not None and model_y_score is not None:
        y_true_opt = np.asarray(model_y_true)
        y_score_opt = np.asarray(model_y_score)

        if y_true_opt.shape[0] != y_score_opt.shape[0]:
            print(
                'Warning: model payload Y/y_score length mismatch; '
                'falling back to predict input records for threshold optimization.'
            )
            y_true_opt = None
            y_score_opt = None

    if y_true_opt is None or y_score_opt is None:
        y_true_opt = np.asarray(input_y_true)
        y_score_opt = np.asarray(input_y_score)

    if method == 'f1':
        threshold = _optimize_threshold_f1_from_pr_curve(y_true_opt, y_score_opt)
    else:
        raise ValueError(
            f'Unsupported predict.model.threshold_selection={method!r}. '
            'Supported values: null, "F1" (or "f1").'
        )

    cache[cache_key] = float(threshold)
    return float(threshold)


def _optimize_threshold_f1_from_pr_curve(y_true, y_score):
    """
    Optimize threshold by maximizing F1 derived from PR-curve points.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if np.unique(y_true).size != 2:
        raise ValueError(
            'F1 threshold optimization requires binary ground-truth labels in predict input records.'
        )

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if thresholds.size == 0:
        return 0.5

    pr = precision[:-1]
    rc = recall[:-1]
    denom = pr + rc
    f1_vals = np.where(denom > 0, 2.0 * pr * rc / denom, 0.0)
    best_idx = int(np.nanargmax(f1_vals))
    best_threshold = float(thresholds[best_idx])
    return _clip_threshold_open_interval(best_threshold)


def _predicted_label_probability_confidence(pos_proba, y_pred):
    """Return confidence as probability that the predicted label is correct."""
    pos_proba = np.asarray(pos_proba, dtype=float)
    y_pred = np.asarray(y_pred)

    if pos_proba.shape[0] != y_pred.shape[0]:
        raise ValueError(
            'Confidence calculation requires pos_proba and y_pred to have the same length.'
        )

    if np.any(~np.isfinite(pos_proba)):
        raise ValueError('Confidence calculation received non-finite positive-class probabilities.')

    if np.any((pos_proba < 0.0) | (pos_proba > 1.0)):
        raise ValueError('Confidence calculation requires probabilities within [0, 1].')

    confidence = np.where(y_pred == 1, pos_proba, 1.0 - pos_proba)
    return np.clip(confidence, 0.0, 1.0)


# retired: kept for reference only; replaced by
# _predicted_label_probability_confidence().
def _normalized_threshold_confidence(pos_proba, threshold):
    """
    Compute normalized distance from the threshold, or decision confidence
    in [-1, 1] interval based on positive label probability and the decision
    threshold.

    The expression is
    C = (P - T)/|B - T|, with B = 0 if P < T and B = 1 if P > T

    Thus confidence (normalized distance) will be negative for normal labels,
    and positive for abnormal labels.

    """
    threshold = _clip_threshold_open_interval(float(threshold))
    pos_proba = np.asarray(pos_proba, dtype=float)

    above = pos_proba > threshold
    conf = np.empty_like(pos_proba, dtype=float)
    conf[above] = (pos_proba[above] - threshold) / (1.0 - threshold)
    conf[~above] = (pos_proba[~above] - threshold) / threshold
    return np.clip(conf, -1.0, 1.0)

def _clip_threshold_open_interval(threshold, eps=1e-6):
    """Clip threshold to open interval (0, 1) to avoid division-by-zero."""
    return float(np.clip(threshold, eps, 1.0 - eps))


def add_joint_predictions(out_df, joint_prediction_cfg, ss, model_cfg):
    """Add cross-validated joint predictions to summarize-only output tables.

        The summarize-only output table is built from per-physician records already
        aligned on ``ScanID``. This helper performs nested stratified CV to avoid
        score calibration and threshold-selection leakage:

        - Outer K-fold loop: train/test split over all records.
        - Inner M-fold loop on each outer-train set: generate OOF scores for all
            rows in that outer-train partition.
        - Fit calibrator and optimize threshold using only those inner OOF scores.
        - Refit joint model on the full outer-train partition and evaluate on the
            corresponding outer-test fold using calibrated scores + selected threshold.

        Repeating this for all outer folds yields out-of-fold joint predictions for
        every record in ``out_df``.

    When ``predict.summarize.joint_prediction.show_pr_curve`` is true (or absent,
    which defaults to true), the PR curve from all combined out-of-fold
    predictions is plotted and saved to a PNG file under ``<out_root>/plt/``.

    Args:
        out_df(pd.DataFrame): summarize-only output table; modified in place.
        joint_prediction_cfg(dict): configuration from
            ``predict.summarize.joint_prediction``.
        ss(obj): reference to the app object used for output path resolution.
        model_cfg(dict): resolved summarize model configuration; used for the
            target-label annotation on the PR curve plot.

    Returns:
        tuple[str, float]: summary-stat row name and mean outer-fold threshold
            value to append to the existing summary statistics block.
    """
    if not isinstance(joint_prediction_cfg, dict) or not joint_prediction_cfg:
        raise ValueError('predict.summarize.joint_prediction must be configured.')

    required_columns = ['TrueLabel', 'JointLabel', 'JointConfidence']
    for column in required_columns:
        if column not in out_df.columns:
            out_df[column] = np.nan

    # Use the complete record set for splitting, regardless of the confidence
    # flag. This keeps the joint threshold estimation aligned with the full
    # summarize-only dataset instead of the confidence-filtered subset.
    labels = pd.to_numeric(out_df['TrueLabel'], errors='coerce')

    if labels.isna().any():
        raise ValueError('TrueLabel contains missing values and cannot be used for joint splitting.')

    labels = labels.astype(int).to_numpy()

    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError('TrueLabel must contain only 0 and 1 values for joint prediction.')

    n_splits = int(joint_prediction_cfg.get('n_splits', 0))
    n_inner_splits = int(joint_prediction_cfg.get('n_inner_splits', 0))

    if n_splits < 2:
        raise ValueError('predict.summarize.joint_prediction.n_splits must be at least 2.')

    if n_inner_splits < 2:
        raise ValueError('predict.summarize.joint_prediction.n_inner_splits must be at least 2.')

    class_counts = np.bincount(labels, minlength=2)     # numpy.bincount() counts the frequency of each value
                                                        # in a 1D array of non-negative integers.  

    if np.any(class_counts < n_splits):
        raise ValueError(
            'predict.summarize.joint_prediction.n_splits is too large for the available class counts. '
            f'Class counts={class_counts.tolist()}, n_splits={n_splits}.'
        )

    # Summarize-only must be reproducible across runs, so always use the
    # top-level project seed rather than any local joint-prediction override.
    seed = ss.args.get('seed', None)

    if seed is None:
        raise ValueError('Top-level ss.args["seed"] must be set for summarize-only joint prediction.')

    seed = int(seed)

    raw_threshold_selection = joint_prediction_cfg.get('threshold_selection', 'f1')

    if raw_threshold_selection is None:
        threshold_selection = 'f1'
    else:
        if not isinstance(raw_threshold_selection, str):
            raise ValueError(
                'predict.summarize.joint_prediction.threshold_selection must be null or a string.'
            )

        method = raw_threshold_selection.strip().lower()

        if method in ('', 'none', 'default'):
            threshold_selection = None
        elif method == 'f1':
            threshold_selection = 'f1'
        else:
            raise ValueError(
                'Unsupported predict.summarize.joint_prediction.threshold_selection value. '
                'Supported values are null and "f1".'
            )

    # Collect all physician-level prediction columns once so the same matrix can
    # be sliced for the held-out fold and the remaining training rows.
    physician_cols = [col for col in out_df.columns if col in {'TrueLabel', 'ScanID', 'Confident'}]
    model_physicians = [
        col for col in out_df.columns
        if col not in physician_cols and not col.endswith('_PredConfidence')
        and col not in {'Mean', 'AbsErr', 'STD', 'MeanConfidence', 'STDConfidence'}
        and not col.startswith('Joint')
    ]   # this is just a list of all available physician names

    if not model_physicians:
        raise ValueError('No physician prediction columns found for joint prediction.')

    confidence_cols = [f'{physician_name}_PredConfidence' for physician_name in model_physicians]

    missing_confidence_cols = [col for col in confidence_cols if col not in out_df.columns]

    if missing_confidence_cols:
        raise ValueError(
            'Missing physician confidence columns required for joint prediction: '
            f'{missing_confidence_cols}'
        )

    # Those are columns to be used for joint prediction, i.e.
    # ['Sophia', 'Maria', 'Eleni', 'Zoe', 'Athina',
    # 'Sophia_PredConfidence', 'Maria_PredConfidence', 'Eleni_PredConfidence', 'Zoe_PredConfidence', 'Athina_PredConfidence']
    # Under phys names are predicted hard labels, under confidences - well, confidences
    required_joint_cols = model_physicians + confidence_cols

    # A sub-frame used for joint prediction
    joint_input_frame = out_df[required_joint_cols]

    if joint_input_frame.isna().any().any():
        missing_cols = joint_input_frame.columns[joint_input_frame.isna().any()].tolist()
        raise ValueError(
            'Joint prediction requires complete physician label/confidence values for every record. '
            f'Missing values were found in columns: {missing_cols}'
        )

    # y, y_proba are nrec x nphysicians predicted labels and their confidences.
    y = out_df[model_physicians].to_numpy(dtype=int)
    y_proba = out_df[confidence_cols].to_numpy(dtype=float)     # Mind that y_proba here is CONFIDENCE
                                                                # not class 1 score

    # This is the outer folds stratifier
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # The flag defaults to True so the curve is shown unless explicitly disabled.
    show_pr_curve = bool(joint_prediction_cfg.get('show_pr_curve', True))

    # Prepare PR curve full pathname
    prcurve_out_dir = _predict_results_root(ss) / 'plt'
    prcurve_out_dir.mkdir(parents=True, exist_ok=True)

    physician_token_str = physician_token(model_physicians) # Smth like 'Sophia_Maria_Eleni_Zoe_Athina'
    target_label = str(model_cfg.get('target_label', ''))
    label_suffix = f'_{target_label}' if target_label else ''
    prcurve_outfname = prcurve_out_dir / f'prcrv_joint_{physician_token_str}{label_suffix}.png'

    # First: nested-CV train/score in out-of-fold manner for every outer split.
    joint_method = joint_prediction_cfg.get('method', 'bayes')
    calibrate_scores = bool(joint_prediction_cfg.get('calibrate_scores', True))

    if joint_method == 'joint_LR':
        joint_lr_cfg = dict(joint_prediction_cfg.get('joint_LR', {}) or {})
    else:
        joint_lr_cfg = {}

    n_records = labels.shape[0]

    # Prepare joint model output arrays
    oof_joint_labels = np.full(n_records, -1, dtype=int)            # Predicted joint labels
    oof_joint_confidence = np.full(n_records, np.nan, dtype=float)  # Predicted joint scores
    fold_thresholds = []                                            # Threshold selected for each outer fold

    # the CV outer loop starts here
    for fold_idx, (train_index, test_index) in enumerate(
        skf.split(np.zeros((labels.shape[0], 1)), labels),  # np.zeros((labels.shape[0], 1)) is just a dummy for skf.split()
        start=1,                                            # because it expects features as 1st arg
    ):                                                      # fold_idx is 1-BASED!!! (start = 1) - cant use 0 as we are adding this
                                                            # to the seed below

        train_index = np.asarray(train_index, dtype=int)
        test_index = np.asarray(test_index, dtype=int)

        outer_train_labels = labels[train_index]
        outer_train_class_counts = np.bincount(outer_train_labels, minlength=2)

        if np.any(outer_train_class_counts < n_inner_splits):
            raise ValueError(
                'predict.summarize.joint_prediction.n_inner_splits is too large for '
                f'outer fold {fold_idx} train class counts. '
                f'Outer-train class counts={outer_train_class_counts.tolist()}, '
                f'n_inner_splits={n_inner_splits}.'
            )

        # Construct inner SKF object for current outer fold
        inner_skf = StratifiedKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=seed + fold_idx,
        )

        # This will contain OOF joint scores for the whole training part of current
        # outer fold when inner loop is done
        outer_train_oof_scores = np.full(train_index.shape[0], np.nan, dtype=float)

        # This is the inner loop starting here...
        for inner_fold_idx, (inner_train_rel, inner_val_rel) in enumerate(
            inner_skf.split(np.zeros((outer_train_labels.shape[0], 1)), outer_train_labels),
            start=1,
        ):
            # The splitter yields the indicies INSIDE the training part of the outer fold
            # Convert them to the global indicies of our records (used in outer loop) 
            inner_train_index = train_index[np.asarray(inner_train_rel, dtype=int)]
            inner_val_index = train_index[np.asarray(inner_val_rel, dtype=int)]

            # Recalculate class priors over training part of inner fold
            inner_priors = np.asarray(
                [
                    np.mean(labels[inner_train_index] == 0),
                    np.mean(labels[inner_train_index] == 1),
                ],
                dtype=float,
            )

            # Now get the joint scores for the VALIDATION part of the
            # inner fold:
            if joint_method == 'bayes':
                inner_val_scores = joint_train_bayes(
                    y=y,
                    y_proba=y_proba,
                    labels=labels,
                    train_index=inner_train_index,
                    score_index=inner_val_index,
                    priors=inner_priors,
                )
            elif joint_method == 'joint_LR':
                inner_val_scores = joint_train_LR(
                    y=y,
                    y_proba=y_proba,
                    labels=labels,
                    train_index=inner_train_index,
                    score_index=inner_val_index,
                    seed=seed + 1000 * fold_idx + inner_fold_idx,
                    priors=inner_priors,
                    joint_lr_cfg=joint_lr_cfg,
                )
            else:
                raise NotImplemented(f'Unsupported joint prediction method {joint_method} specified')

            # Save the scores for the validation part of the inner fold (using the "inner fold" indices
            # as the target indices because those enumerate outer_train_oof_scores array)
            outer_train_oof_scores[np.asarray(inner_val_rel, dtype=int)] = inner_val_scores

            # -------- inner loop done ---------- #

        # We now have OOF scores for the training part of the outer fold saved in the 
        # outer_train_oof_scores array

        if np.any(~np.isfinite(outer_train_oof_scores)):
            raise ValueError(
                f'Inner OOF score construction failed for outer fold {fold_idx}: '
                'non-finite scores detected.'
            )

        # With OOF scores obtained, we can now use the whole training part of current outer fold
        # to train the LR model.
        # Recalculate the priors again (now using the whole outer train part)
        outer_priors = np.asarray(
            [
                np.mean(labels[train_index] == 0),
                np.mean(labels[train_index] == 1),
            ],
            dtype=float,
        )

        # Fit the joint model on the whole outer train fold and get scores
        # for the outer test fold
        if joint_method == 'bayes':
            outer_test_scores_raw = joint_train_bayes(
                y=y,
                y_proba=y_proba,
                labels=labels,
                train_index=train_index,
                score_index=test_index,
                priors=outer_priors,
            )
        elif joint_method == 'joint_LR':
            outer_test_scores_raw = joint_train_LR(
                y=y,
                y_proba=y_proba,
                labels=labels,
                train_index=train_index,
                score_index=test_index,
                seed=seed + 1000 * fold_idx,
                priors=outer_priors,
                joint_lr_cfg=joint_lr_cfg,
            )
        else:
            raise NotImplemented(f'Unsupported joint prediction method {joint_method} specified')

        # Now we have uncalibrated joint scores for the test part of the outer fold in the array
        # outer_test_scores_raw

        # If calibration of the joint model is requested - use the OOF scores of the training part
        # of the current outer fold; then apply to the test part scores  
        if calibrate_scores:
            calibrator, outer_train_scores_for_threshold = cls_calibrate.calibrate_with_method(
                method_name='betacal',
                y_score=outer_train_oof_scores,
                y_true=labels[train_index],
                calibrate_cfg={},
                strict=True,
            )
            outer_test_scores = np.asarray(
                cls_calibrate.apply_calibrator(
                    calibration_obj=calibrator,
                    y_score=outer_test_scores_raw,
                    method='betacal',
                ),
                dtype=float,
            ).ravel()
        else:
            outer_train_scores_for_threshold = outer_train_oof_scores
            outer_test_scores = np.asarray(outer_test_scores_raw, dtype=float)

        # At this point,  we got the final (calibrated or not) scores for the
        # outer test fold in this array: outer_test_scores
        # We also have (calibrated or not) OOF scores in the train part to be used
        # for threshold selection in the array: outer_train_scores_for_threshold

        # Now choose the threshold using the OOF scores of the train part.
        if threshold_selection == 'f1':
            threshold = _optimize_threshold_f1_from_pr_curve(
                labels[train_index],
                outer_train_scores_for_threshold,
            )
        else:
            threshold = 0.5

        # "treshold" now contains the joint threshold value for current outer fold
        # Use it to get HARD LABELS and CONFIDENCES of the test part.
        outer_test_labels = (outer_test_scores > threshold).astype(int)
        outer_test_conf = np.where(
            outer_test_labels == 1,
            outer_test_scores,
            1.0 - outer_test_scores,
        )

        oof_joint_labels[test_index] = outer_test_labels
        oof_joint_confidence[test_index] = outer_test_conf
        fold_thresholds.append(float(threshold))

        print(
            f'Joint {joint_method} nested-CV outer fold {fold_idx}/{n_splits}: '
            f'threshold={float(threshold):.4f}, n_train={len(train_index)}, n_test={len(test_index)}'
        )

        # ------ the CV outer loop done ------- 

    if not fold_thresholds:
        raise ValueError('Joint prediction produced no CV folds.')

    if np.any(oof_joint_labels < 0) or np.any(~np.isfinite(oof_joint_confidence)):
        raise ValueError('Joint nested-CV prediction failed to populate all out-of-fold rows.')

    out_df['JointLabel'] = oof_joint_labels
    out_df['JointConfidence'] = oof_joint_confidence

    # Only mean value of the fold thresholds will be saved to the CSV as the "joint threshold"
    threshold = float(np.mean(fold_thresholds))
    threshold_summary_name = 'joint_threshold_F1' if threshold_selection == 'f1' else 'joint_threshold_0.5'

    if show_pr_curve:
        y_true_num = pd.to_numeric(out_df['TrueLabel'], errors='coerce')
        y_joint_num = pd.to_numeric(out_df['JointLabel'], errors='coerce')
        y_conf_num = pd.to_numeric(out_df['JointConfidence'], errors='coerce')
        eval_mask = (
            y_true_num.isin([0, 1])
            & y_joint_num.isin([0, 1])
            & y_conf_num.notna()
            & y_conf_num.between(0.0, 1.0)
        )

        if eval_mask.any():
            y_true_eval = y_true_num.loc[eval_mask].to_numpy(dtype=int)
            y_joint_eval = y_joint_num.loc[eval_mask].to_numpy(dtype=int)
            y_conf_eval = y_conf_num.loc[eval_mask].to_numpy(dtype=float)
            y_pos_score_eval = np.where(y_joint_eval == 1, y_conf_eval, 1.0 - y_conf_eval)

            plot_and_save_pr_curve(
                y_true_eval,
                y_pos_score_eval,
                step_name='joint',
                outfname=prcurve_outfname,
                show_plot=show_pr_curve,
                target_label=target_label if target_label else None,
                physician=model_physicians,
            )
            print(
                'Joint prediction PR curve (out-of-fold combined predictions) '
                f'saved to: {prcurve_outfname}'
            )
        else:
            print('Skipping joint PR curve plot: no valid out-of-fold rows were found.')

    # Return a summary-stat entry so the threshold used for the joint
    # prediction logic is recorded alongside the other summarize-only metrics.
    return threshold_summary_name, threshold

def joint_train_bayes(
    y,
    y_proba,
    labels,
    train_index,
    score_index,
    priors,
):
    """Fit ``JointBayes`` and return raw class-1 scores for ``score_index`` rows.

    Args:
        y (np.ndarray): Per-physician hard predictions for all rows.
        y_proba (np.ndarray): Per-physician confidence values for all rows,
            where each value is the probability that the corresponding hard
            predicted label is correct (not the raw class-1 probability).
        labels (np.ndarray): Ground-truth binary labels for all rows.
        train_index (np.ndarray): Indices used for fold training.
        score_index (np.ndarray): Indices to score with the fitted fold model.
        priors (array-like): Class priors forwarded to ``JointBayes``.

    Returns:
        np.ndarray: Raw class-1 probabilities for rows in ``score_index``.
    """
    train_index = np.asarray(train_index, dtype=int)
    score_index = np.asarray(score_index, dtype=int)

    joint_model = JointBayes(
        threshold_selection=None,
        priors=priors,
    )

    joint_model.fit(
        [y[train_index, :], y_proba[train_index, :]],
        labels[train_index],
    )
    score_probs = np.asarray(
        joint_model.predict_proba([y[score_index, :], y_proba[score_index, :]]),
        dtype=float,
    )
    return score_probs[:, 1]


def joint_train_LR(
    y,
    y_proba,
    labels,
    train_index,
    score_index,
    seed,
    priors,
    joint_lr_cfg=None,
):
    """Fit ``JointLR`` and return raw class-1 scores for ``score_index`` rows.

    Args:
        y (np.ndarray): Per-physician hard predictions for all rows.
        y_proba (np.ndarray): Per-physician confidence values for all rows,
            where each value is the probability that the corresponding hard
            predicted label is correct (not the raw class-1 probability).
        labels (np.ndarray): Ground-truth binary labels for all rows.
        train_index (np.ndarray): Indices used for fold training.
        score_index (np.ndarray): Indices to score with the fitted fold model.
        seed (int): Global seed for deterministic splitting/config defaults.
        priors (array-like): Global class priors; also used to derive balanced
            class weights when requested.
        joint_lr_cfg (dict | None): Optional JointLR constructor configuration.

    Returns:
        np.ndarray: Raw class-1 probabilities for rows in ``score_index``.
    """
    joint_lr_cfg = {} if joint_lr_cfg is None else dict(joint_lr_cfg)
    train_index = np.asarray(train_index, dtype=int)
    score_index = np.asarray(score_index, dtype=int)

    # Make JointLR deterministic by default with the global seed from JSON.
    if joint_lr_cfg.get('random_state', None) is None:
        joint_lr_cfg['random_state'] = int(seed)

    # Balance the class weights in accordance with global priors
    if joint_lr_cfg.get('class_weight') == 'balanced':
        priors = np.asarray(priors, dtype=float)

        if priors.shape != (2,):
            raise ValueError('priors must contain exactly two class probabilities for JointLR.')

        if np.any(~np.isfinite(priors)) or np.any(priors <= 0.0):
            raise ValueError('priors must contain finite positive probabilities for JointLR.')

        prior_sum = float(np.sum(priors))

        if prior_sum <= 0.0:
            raise ValueError('priors must sum to a positive value for JointLR.')

        priors = priors / prior_sum
        joint_lr_cfg['class_weight'] = {
            0: float(1.0 / (2.0 * priors[0])),
            1: float(1.0 / (2.0 * priors[1])),
        }

    joint_model = JointLR(joint_LR=joint_lr_cfg)

    joint_model.fit(
        [y[train_index, :], y_proba[train_index, :]],
        labels[train_index],
    )

    score_probs = np.asarray(
        joint_model.predict_proba([y[score_index, :], y_proba[score_index, :]]),
        dtype=float,
    )
    return score_probs[:, 1]

