"""
Compute model calibration diagnostics from saved out-of-fold probabilities.
"""
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

STEP = 'calibrate'
SUPPORTED_CALIBRATION_METHODS = ('platt', 'betacal', 'isotonic')
SUPPORTED_CALIBRATION_METHODS_WITH_BEST = SUPPORTED_CALIBRATION_METHODS + ('best',)

# The "dummy" calibrator Briar score formula
dummy_BS = lambda p: p*(1-p)

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


def calibrate_with_method(method_name, y_score, y_true, calibrate_cfg, strict=True):
    """
    Fit a calibrator by method and return calibrated probabilities.

    Args:
        method_name (str): Calibration method name. Supported values are
            ``'platt'``, ``'betacal'``, and ``'isotonic'``.
        y_score (np.ndarray): Uncalibrated positive-class probabilities,
            shape ``(n_samples,)``.
        y_true (np.ndarray): Binary labels encoded as 0/1,
            shape ``(n_samples,)``.
        calibrate_cfg (dict): Calibration config dict. Per-method constructor
            overrides can be passed under keys ``'platt'``, ``'betacal'``, and
            ``'isotonic'``.
        strict (bool): If ``True``, missing optional dependencies raise.
            If ``False``, methods with missing optional dependencies return
            ``(None, None)``.

    Returns:
        tuple:
            - fitted calibrator object
            - 1D ``np.ndarray`` with calibrated probabilities

        Returns ``(None, None)`` only when ``strict=False`` and an optional
        dependency is unavailable.
    """
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


def apply_calibrator(calibration_obj=None, y_score=None, method=None):
    """
    Apply a fitted calibrator to uncalibrated probabilities.

    Args:
        calibration_obj (dict | object | None): Either a legacy calibration payload
            dictionary with ``'method'`` and ``'calibrator'`` keys, or a fitted
            calibrator object.
        y_score (np.ndarray): Uncalibrated positive-class probabilities,
            shape ``(n_samples,)``.
        method (str | None): Calibration method name. Used when ``calibration_obj``
            is a fitted calibrator object rather than a legacy payload dictionary.

    Returns:
        np.ndarray: 1D array of calibrated probabilities,
        shape ``(n_samples,)``.
    """
    if calibration_obj is None:
        raise ValueError('calibration_obj is required to apply calibration.')

    fitted = calibration_obj
    stored_method = method

    if isinstance(calibration_obj, dict):
        stored_method = calibration_obj['method'] if method is None else method
        fitted = calibration_obj['calibrator']

    if stored_method is None:
        if hasattr(fitted, 'predict_proba'):
            stored_method = 'platt'
        elif hasattr(fitted, 'predict'):
            stored_method = 'betacal'
        else:
            raise ValueError('Could not infer calibration method from calibrator object.')

    if stored_method == 'platt':
        return fitted.predict_proba(y_score.reshape(-1, 1))[:, 1]

    if stored_method in ('betacal', 'isotonic'):
        return np.asarray(fitted.predict(y_score)).ravel()

    raise ValueError(
        f'Cannot re-apply stored calibration method={stored_method!r}: unknown method.'
    )


# Backward-compatible aliases for previous private helper names.
def _calibrate_with_method(method_name, y_score, y_true, calibrate_cfg, strict=True):
    """Compatibility wrapper for ``calibrate_with_method``."""
    return calibrate_with_method(method_name, y_score, y_true, calibrate_cfg, strict=strict)


def _apply_calibrator(calibration_obj, y_score):
    """Compatibility wrapper for ``apply_calibrator``."""
    return apply_calibrator(calibration_obj, y_score)


def cls_calibrate(ss):
    """
    Load saved true labels and out-of-fold probabilities from a model pickle,
    then compute and print calibration-related summary metrics.

    Pickle resolution follows the same precedence as ``cls_predict``:

    - If top-level ``pickle`` is non-null, use it (relative to ``out_root``).
    - Otherwise derive pickle path from ``calibrate.model`` settings.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing
    """
    from cls_predict import _resolve_model_pickle_path

    calibrate_cfg = ss.args.get('calibrate', {}) or {}
    pkl_pname = _resolve_model_pickle_path(ss, calibrate_cfg, step_name='calibrate')

    print(f'Loading model payload for calibration from: {pkl_pname}')

    if not pkl_pname.exists():
        raise FileNotFoundError(f'Model pickle not found: {pkl_pname}')

    with open(pkl_pname, 'rb') as fp:
        payload = pickle.load(fp)

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

    neg_count = int(np.count_nonzero(y_true == 0))
    pos_count = int(np.count_nonzero(y_true == 1))

    if neg_count == 0:
        raise ValueError('Cannot compute positive/negative ratio: count of negative labels is zero.')

    pos_neg_ratio = pos_count / float(neg_count)
    pos_ratio = pos_count / float(len(y_true))
    brier_uncalibrated = brier_score_loss(y_true, y_score)

    method = str(calibrate_cfg.get('method', 'platt')).strip().lower()
    requested_method = method

    if not method:
        raise ValueError('calibrate.method cannot be empty.')

    # ------------------------------------------------------------------
    # Decide whether to (re-)fit or reuse an already stored calibrator
    # ------------------------------------------------------------------
    force_recalibrate = bool(calibrate_cfg.get('force_recalibrate', False))
    stored_calibration = payload.get('calibration')

    if stored_calibration is not None and not force_recalibrate:
        stored_method_name = stored_calibration.get('method', '<unknown>')
        print(
            f'Reusing existing calibrator ({stored_method_name}) from pickle. '
            'Set calibrate.force_recalibrate=true to override.'
        )
        method = stored_method_name
        fitted_calibrator = stored_calibration['calibrator']
        y_score_corrected = apply_calibrator(stored_calibration, y_score)
        save_calibration = False
    else:
        if stored_calibration is not None:
            print('force_recalibrate=true: re-fitting calibrator.')

        if method == 'best':
            candidate_methods = SUPPORTED_CALIBRATION_METHODS
            best_method = None
            best_score = None
            best_y_score = None
            best_fitted = None

            print('Evaluating calibration methods for calibrate.method="best":')
            for candidate in candidate_methods:
                fitted_c, y_candidate = calibrate_with_method(
                    candidate,
                    y_score,
                    y_true,
                    calibrate_cfg,
                    strict=False,
                )

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
            method = best_method
            fitted_calibrator = best_fitted
            y_score_corrected = best_y_score
        else:
            fitted_calibrator, y_score_corrected = calibrate_with_method(
                method,
                y_score,
                y_true,
                calibrate_cfg,
                strict=True,
            )

        save_calibration = True

    if np.any(~np.isfinite(y_score_corrected)):
        raise ValueError('Calibrated probabilities contain non-finite values.')

    if np.any((y_score_corrected < 0.0) | (y_score_corrected > 1.0)):
        raise ValueError('Calibrated probabilities must be within [0, 1].')

    brier_corrected = brier_score_loss(y_true, y_score_corrected)

    # ------------------------------------------------------------------
    # Persist calibrator back to pickle when newly fitted
    # ------------------------------------------------------------------
    if save_calibration:
        payload['calibration'] = {
            'method': method,
            'calibrator': fitted_calibrator,
            'brier_uncalibrated': brier_uncalibrated,
            'brier_corrected': brier_corrected,
        }
        with open(pkl_pname, 'wb') as fp:
            pickle.dump(payload, fp)
        print(f'Calibrator ({method}) saved to pickle: {pkl_pname}')

    if requested_method == 'best':
        print(f'\nOptimal method chosen: {method}')
    else:
        print(f'Calibration method: {method}')

    print(f'A-priori probability of positive label (Y==1 / nY): {pos_ratio:.4e}')
    print(f'Dummy classifier Brier score: {dummy_BS(pos_ratio):.4e}')
    print(f'Uncalibrated classifier Brier score: {brier_uncalibrated:.4e}')
    print(f'Calibrated classifier Brier score: {brier_corrected:.4e}')
    print(f'\nStep *{STEP}* completed successfully')

