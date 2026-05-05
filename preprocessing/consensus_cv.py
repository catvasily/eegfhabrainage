"""
**Consensus nested-CV utilities with reusable feature-selection reducer.**
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def consensus_cv(
    estimator,
    raw_features,
    y,
    outer_cv_splitter,
    consensus_cfg,
    standardize=False,
    seed=None,
):
    """
    Run consensus nested CV.

    Per outer fold this function:
    1) builds a consensus reducer on outer-train using inner selection folds,
    2) optionally tunes hyperparameters on reduced outer-train,
    3) fits the final fold model and evaluates on outer-test.

    Args:
        estimator: sklearn-compatible classifier. Must implement ``fit`` and
            ``predict``. If ``predict_proba`` is implemented, probability-based
            outputs are populated.
        raw_features (ndarray): Raw feature tensor with shape
            ``(n_samples, n_chans, n_freqs, n_parms)``.
        y (ndarray): Label vector of shape ``(n_samples,)``.
        outer_cv_splitter: Outer CV splitter (for example ``StratifiedKFold``)
            that yields outer train/test indices via ``split(raw_features, y)``.
        consensus_cfg (dict): Settings block under JSON key ``consensus_cv``.
            Common keys include ``inner_n_splits``, ``selection_method``,
            ``f_classif_top_k``, ``top_k``, ``min_score``, ``n_jobs``, and
            optional ``grid_tuning`` settings such as ``enabled``, ``n_splits``,
            ``param_grid``, ``scoring``, ``n_jobs``, and ``verbose``.
        standardize (bool): If True, fit/apply ``StandardScaler`` on selected
            features inside each outer fold (fit on outer-train only).
        seed (int|None): Random seed used for inner splitters and deterministic
            behavior where applicable.

    Returns:
        outer_scores (ndarray): Per-outer-fold test accuracy, shape
            ``(n_outer_folds,)``.
        y_pred_all (ndarray): Out-of-fold predicted class labels, shape
            ``(n_samples,)``.
        y_score_all (ndarray): Out-of-fold positive-class score used for PR/AUC,
            shape ``(n_samples,)``. When probabilities are unavailable for a fold,
            corresponding positions are zeros.
        y_proba_all (ndarray|None): Out-of-fold class probabilities, shape
            ``(n_samples, n_classes)``, or ``None`` if estimator does not provide
            probabilities.
        fold_summaries (list[dict]): Per-outer-fold metadata entries in outer
            fold order. Each dict includes at least ``fold``, ``n_train``,
            ``n_test``, ``outer_test_accuracy``, ``n_features_selected``, and
            ``selection_method``. When grid tuning is enabled, it also includes
            ``best_params`` and ``best_inner_score``.
    """
    raw_features = np.asarray(raw_features)
    y = np.asarray(y)

    if raw_features.ndim != 4:
        raise ValueError(f'raw_features must be 4D, got shape {raw_features.shape}')
    if y.ndim != 1:
        raise ValueError(f'y must be 1D, got shape {y.shape}')
    if raw_features.shape[0] != y.shape[0]:
        raise ValueError(f'raw_features/y sample mismatch: {raw_features.shape[0]} != {y.shape[0]}')

    n_samples = raw_features.shape[0]
    y_pred_all = np.zeros(n_samples, dtype=float)
    y_score_all = np.zeros(n_samples, dtype=float)
    y_proba_all = None

    outer_scores = []
    fold_summaries = []

    tuning_cfg = consensus_cfg.get('grid_tuning', {}) or {}
    tuning_enabled = bool(tuning_cfg.get('enabled', False))
    tuning_param_grid = tuning_cfg.get('param_grid', None)
    tuning_scoring = tuning_cfg.get('scoring', 'accuracy')
    tuning_n_jobs = int(tuning_cfg.get('n_jobs', 1))
    tuning_verbose = int(tuning_cfg.get('verbose', 0))
    tuning_n_splits = int(tuning_cfg.get('n_splits', int(consensus_cfg.get('inner_n_splits', 5))))

    if tuning_enabled:
        if tuning_n_splits < 2:
            raise ValueError('consensus_cv.grid_tuning.n_splits must be >= 2')
        if not isinstance(tuning_param_grid, (dict, list)) or len(tuning_param_grid) == 0:
            raise ValueError(
                'consensus_cv.grid_tuning.enabled is true, but grid_tuning.param_grid is missing or empty'
            )

    total_outer = outer_cv_splitter.get_n_splits()
    for ifold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(raw_features, y)):
        print(f'  Consensus CV outer fold {ifold + 1}/{total_outer}')

        raw_train = raw_features[train_idx]
        raw_test = raw_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train, reducer, selection_meta = build_and_fit_consensus_reducer(
            raw_train,
            y_train,
            consensus_cfg,
            standardize=standardize,
            seed=seed,
        )

        X_test = apply_consensus_reducer(raw_test, reducer)

        best_inner_score = None
        best_params = None

        if tuning_enabled:
            tune_splitter = StratifiedKFold(
                n_splits=tuning_n_splits,
                shuffle=True,
                random_state=seed,
            )
            grid = GridSearchCV(
                estimator=clone(estimator),
                param_grid=tuning_param_grid,
                scoring=tuning_scoring,
                cv=tune_splitter,
                refit=True,
                n_jobs=tuning_n_jobs,
                verbose=tuning_verbose,
            )
            grid.fit(X_train, y_train)
            fold_estimator = grid.best_estimator_
            best_inner_score = float(grid.best_score_)
            best_params = dict(grid.best_params_)
        else:
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_train, y_train)

        try:
            y_proba_fold = fold_estimator.predict_proba(X_test)
            y_pred_fold = np.argmax(y_proba_fold, axis=1)
            if y_proba_fold.shape[1] > 1:
                y_score_fold = y_proba_fold[:, 1]
            else:
                y_score_fold = y_proba_fold.ravel()
        except Exception:
            y_pred_fold = fold_estimator.predict(X_test)
            y_score_fold = np.zeros(len(y_pred_fold), dtype=float)
            y_proba_fold = None

        fold_acc = accuracy_score(y_test, y_pred_fold)
        outer_scores.append(fold_acc)

        y_pred_all[test_idx] = y_pred_fold
        y_score_all[test_idx] = y_score_fold

        if y_proba_fold is not None:
            if y_proba_all is None:
                y_proba_all = np.zeros((n_samples, y_proba_fold.shape[1]), dtype=float)
            y_proba_all[test_idx, :] = y_proba_fold

        fold_summary = {
            'fold': ifold,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'outer_test_accuracy': float(fold_acc),
            'n_features_selected': int(selection_meta['n_features_selected']),
            'selection_method': selection_meta['selection_method'],
            'selected_idx': reducer['selected_idx'],
            'n_features_in': selection_meta['n_features_in'],
        }

        if best_params is not None:
            fold_summary['best_params'] = best_params
            fold_summary['best_inner_score'] = best_inner_score

        fold_summaries.append(fold_summary)

        print(
            f'    Consensus selected: {selection_meta["n_features_selected"]} features; '
            f'outer-test accuracy: {fold_acc:.4f}'
        )

    return np.asarray(outer_scores), y_pred_all, y_score_all, y_proba_all, fold_summaries

def build_and_fit_consensus_reducer(raw_features, y, consensus_cfg, standardize=False, seed=None):
    """
    Fit a consensus feature-selection reducer on raw 4D features.

    The reducer is produced by running feature selection on K inner folds and
    intersecting selected feature sets across all folds.

    Args:
        raw_features (ndarray): shape (n_samples, n_chans, n_freqs, n_parms)
        y (ndarray): shape (n_samples,)
        consensus_cfg (dict): consensus_cv config block from JSON
        standardize (bool): apply StandardScaler after feature selection
        seed (int|None): RNG seed for splitters and MI estimator

    Returns:
        X_reduced (ndarray): Reduced matrix with shape
            ``(n_samples, n_selected)``.
        reducer (dict): Serializable reducer payload for later reuse with keys
            including ``method``, ``pipeline``, ``selected_idx``,
            ``n_features_in``, ``selection_method``, and ``inner_n_splits``.
        selection_meta (dict): Selection diagnostics with keys including
            ``inner_n_splits``, ``selection_method``, ``fold_details``,
            ``n_features_in``, and ``n_features_selected``.
    """
    X = raw_features.reshape(raw_features.shape[0], -1)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f'X/y sample mismatch: {X.shape[0]} != {y.shape[0]}')

    inner_n_splits = int(consensus_cfg.get('inner_n_splits', 5))

    if inner_n_splits < 2:
        raise ValueError('consensus_cv.inner_n_splits must be >= 2')

    inner_splitter = StratifiedKFold(
        n_splits=inner_n_splits,
        shuffle=True,
        random_state=seed,
    )

    selected_idx, fold_details = _compute_consensus_selected_indices(
        X,
        y,
        inner_splitter,
        consensus_cfg,
        seed,
    )

    # The data returned by _compute_consensus_selected_indices() is:
    #   selected_idx - sorted 1D array of final selected feature idx
    #   fold_details (list[dict]): per-inner-fold metadata

    X_selected = X[:, selected_idx]     # These are reduced dim flat features

    if standardize:
        pipeline = make_pipeline(StandardScaler())
        X_reduced = pipeline.fit_transform(X_selected)
    else:
        pipeline = None
        X_reduced = X_selected

    reducer = {
        'method': 'consensus_cv',
        'pipeline': pipeline,
        'selected_idx': selected_idx,
        'n_features_in': int(X.shape[1]),       # Save orig feature dim for verif when reducer is applied
        'selection_method': str(consensus_cfg.get('selection_method', 'mutual_info')).lower(),
        'inner_n_splits': inner_n_splits,
    }

    selection_meta = {
        'inner_n_splits': inner_n_splits,
        'selection_method': reducer['selection_method'],
        'fold_details': fold_details,
        'n_features_in': int(X.shape[1]),
        'n_features_selected': int(selected_idx.shape[0]),
    }

    return X_reduced, reducer, selection_meta

def apply_consensus_reducer(raw_features, reducer):
    """
    Apply a fitted consensus reducer to raw 4D features.
    """
    X = raw_features.reshape(raw_features.shape[0], -1)
    selected_idx = np.asarray(reducer.get('selected_idx', []), dtype=int)

    if selected_idx.size == 0:
        raise ValueError('Consensus reducer has no selected_idx; cannot transform data')

    n_features_in = reducer.get('n_features_in', None)

    if n_features_in is not None and int(n_features_in) != X.shape[1]:
        raise ValueError(
            'Consensus reducer feature dimension mismatch: '
            f'transformer expects {int(n_features_in)}, got {X.shape[1]}'
        )

    X_selected = X[:, selected_idx]

    pipeline = reducer.get('pipeline', None)

    if pipeline is None:
        return X_selected

    return pipeline.transform(X_selected)

def _compute_consensus_selected_indices(X, y, splitter, consensus_cfg, seed):
    """
    Compute consensus-selected feature indices across inner CV folds.

    For each split produced by ``splitter``, the selector is fit on that split's
    fold partition (the second index array returned by ``split``), producing one
    set of selected feature indices per fold. Consensus is then computed as the
    strict intersection across all ``k`` fold-partition selections.

    Args:
        X (ndarray): 2D feature matrix of shape ``(n_samples, n_features)``.
        y (ndarray): 1D label vector of shape ``(n_samples,)``.
        splitter: CV splitter instance (for example ``StratifiedKFold``)
            implementing ``split(X, y)`` and yielding
            ``(complement_idx, inner_fold_idx)``. For each split, ``complement_idx``
            is used to fit ``f_classif`` preselection, then both subsets are
            reduced to those preselected features before the configured
            ``selection_method`` is applied on ``inner_fold_idx``.
        consensus_cfg (dict): Consensus settings dictionary. Uses keys such as
            ``selection_method``, ``f_classif_top_k``, ``top_k``, ``min_score``,
            and ``n_jobs``.
        seed (int|None): Random seed propagated to feature selection routines.

    Returns:
        selected_idx (ndarray[int]): Sorted 1D array of final selected feature
            indices in flattened-feature coordinate space.
        fold_details (list[dict]): Per-inner-fold selection metadata in fold
            order. Each entry contains:
            - ``fold`` (int|str): inner fold index, or
              ``"fallback_full_train"`` if intersection is empty
            - ``n_fold_samples`` (int): number of samples used for selection in
                that fold
            - ``n_selected`` (int): number of features selected by that fold
            - ``note`` (str, optional): fallback marker
              ``"intersection_empty_fallback"`` when applicable

    Notes:
        If strict intersection across folds is empty, this function falls back
        to one selection pass on full ``(X, y)`` to keep downstream training
        feasible.
    """
    selected_per_fold = []
    fold_details = []

    method = str(consensus_cfg.get('selection_method', 'mutual_info')).lower()

    if method in ('mutual_entropy', 'mutual_entrophy'):
        method = 'mutual_info'

    inner_splits = list(splitter.split(X, y))
    total_inner = len(inner_splits)
    inner_fold_indices = [inner_fold_idx for _, inner_fold_idx in inner_splits]

    def _process_inner_fold(item):
        ifold, (complement_idx, inner_fold_idx) = item

        # Returned are indicies of most important features in ascending
        # order. Note that because indicies are sorted, corresponding
        # importances are no longer sorted in order of descending scores.
        preselected_idx = _preselect_features_f_classif(
            X[complement_idx],
            y[complement_idx],
            consensus_cfg,
            X.shape[1],
        )

        # Special case: if method is 'none', use preselected indices directly
        # without further refinement via _select_features_once()
        if method == 'none':
            fold_selected = preselected_idx
        else:
            X_inner = X[inner_fold_idx][:, preselected_idx]
            y_inner = y[inner_fold_idx]

            # Apply configured selector after fold-wise f_classif preselection.
            fold_selected_reduced = _select_features_once(X_inner, y_inner, method, consensus_cfg, seed)
            fold_selected = preselected_idx[fold_selected_reduced]

        detail = {
            'fold': ifold,
            'n_fold_samples': int(len(inner_fold_idx)),
            'n_preselected': int(preselected_idx.shape[0]),
            'n_selected': int(fold_selected.shape[0]),
        }
        return set(fold_selected.tolist()), detail

    # Inner fold selections are independent: run one thread per inner split.
    with ThreadPoolExecutor(max_workers=total_inner) as pool:
        fold_results = list(pool.map(_process_inner_fold, enumerate(inner_splits)))

    for selected_set, detail in fold_results:
        selected_per_fold.append(selected_set)
        fold_details.append(detail)
        print(
            f'    Inner fold {int(detail["fold"]) + 1}/{total_inner}: '
            f'{int(detail["n_selected"])} features selected '
            f'after f_classif preselection to {int(detail["n_preselected"])} '
            f'(samples: {int(detail["n_fold_samples"])})'
        )

    # Defensive check: the k fold partitions should be a non-overlapping cover
    # of all samples in X (expected for KFold/StratifiedKFold split outputs).
    if len(inner_fold_indices) > 0:
        concatenated = np.concatenate(inner_fold_indices)

        if concatenated.size != X.shape[0] or np.unique(concatenated).size != X.shape[0]:
            raise ValueError(
                'Inner fold partitions do not form a full non-overlapping cover '
                f'of samples: got {concatenated.size} indexed positions for {X.shape[0]} samples'
            )

    if len(selected_per_fold) == 0:
        raise ValueError('No inner folds produced for consensus selection')

    # Choose consensus by simple intersection. Use other strategies
    # right here (if needed)
    consensus_set = set.intersection(*selected_per_fold)

    if len(consensus_set) == 0:
        # Fallback keeps pipeline usable when strict intersection is empty.
        preselected_idx = _preselect_features_f_classif(X, y, consensus_cfg, X.shape[1])
        
        # Special case: if method is 'none', use preselected indices directly
        if method == 'none':
            selected_idx = preselected_idx
        else:
            selected_idx_reduced = _select_features_once(
                X[:, preselected_idx],
                y,
                method,
                consensus_cfg,
                seed,
            )
            # selected_idx_reduced, then selected_idx are indicies of most
            # important features in ascending order. Note that because
            # indicies (not importances) are now sorted, corresponding
            # importances themselves are no longer sorted by descending scores.
            selected_idx = preselected_idx[selected_idx_reduced]
        
        fold_details.append({
            'fold': 'fallback_full_train',
            'n_fold_samples': int(X.shape[0]),
            'n_preselected': int(preselected_idx.shape[0]),
            'n_selected': int(selected_idx.shape[0]),
            'note': 'intersection_empty_fallback',
        })
        return np.sort(selected_idx.astype(int)), fold_details

    selected_idx = np.array(sorted(consensus_set), dtype=int)
    return selected_idx, fold_details

def _select_features_once(X, y, method, consensus_cfg, seed):
    """
    Select feature indices from one data subset using configured score rules.

    Computes a 1D score per feature via ``_compute_feature_scores``, then keeps
    indices passing optional constraints from ``consensus_cfg``:
    - ``min_score``: keep only features with score >= threshold
    - ``top_k``: keep at most the top-k highest-scoring features

    If no index survives filtering, a single fallback feature is kept using the
    maximum available score to avoid returning an empty selection.

    Args:
        X (ndarray): 2D array of shape ``(n_samples, n_features)``.
        y (ndarray): 1D labels of shape ``(n_samples,)``.
        method (str): Feature scoring method (for example ``mutual_info`` or
            ``multisurf``).
        consensus_cfg (dict): Selection config; may include ``top_k`` and
            ``min_score`` and ``n_jobs``.
        seed (int|None): Random seed passed to scoring routines when relevant.

    Returns:
        ndarray[int]: Sorted 1D array of selected feature indices.
    """
    # Calculate a 1D array of n_features feature scores
    scores = _compute_feature_scores(X, y, method, consensus_cfg, seed)

    if scores.ndim != 1 or scores.shape[0] != X.shape[1]:
        raise ValueError(
            f'Feature selector returned invalid score vector shape {scores.shape}; expected ({X.shape[1]},)'
        )

    top_k = consensus_cfg.get('top_k', None)
    min_score = consensus_cfg.get('min_score', None)

    # Filter out infinities, just in case
    valid = np.isfinite(scores)
    idx = np.flatnonzero(valid)     # idx of valid scores

    if min_score is not None:
        min_score = float(min_score)
        idx = idx[scores[idx] >= min_score]

    if top_k is not None:
        top_k = max(1, min(int(top_k), X.shape[1]))

        if idx.size == 0:
            order = np.argsort(scores)[::-1]
            idx = order[:top_k]
        else:
            order = idx[np.argsort(scores[idx])[::-1]]
            idx = order[: min(top_k, order.size)]

    if idx.size == 0:
        best_idx = int(np.nanargmax(scores))
        idx = np.array([best_idx], dtype=int)

    return np.sort(idx.astype(int))

def _preselect_features_f_classif(X_train, y_train, consensus_cfg, n_total_features):
    """
    Preselect top features using ``f_classif`` on one inner-fold train subset.

    The number of retained features is controlled by
    ``consensus_cfg['f_classif_top_k']`` when present. If it is missing or
    ``None``, all features are retained.
    """
    pre_top_k = consensus_cfg.get('f_classif_top_k', None)

    if pre_top_k is None:
        return np.arange(n_total_features, dtype=int)

    pre_top_k = max(1, min(int(pre_top_k), n_total_features))

    # f_classif can produce NaN/inf scores (for constant features, etc.).
    scores, _ = f_classif(X_train, y_train)
    scores = np.asarray(scores)
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)

    if np.all(np.isneginf(safe_scores)):
        return np.arange(pre_top_k, dtype=int)

    order = np.argsort(safe_scores)[::-1]
    return np.sort(order[:pre_top_k].astype(int))

def _compute_feature_scores(X, y, method, consensus_cfg, seed):
    n_jobs = int(consensus_cfg.get('n_jobs', 1))

    if n_jobs == 0:
        raise ValueError('consensus_cv.n_jobs must not be 0')

    if method == 'mutual_info':
        return mutual_info_classif(X, y, random_state=seed)

    if method == 'multisurf':
        try:
            from skrebate import MultiSURF
        except Exception as exc:
            raise ImportError(
                'consensus_cv selection_method="multisurf" requires skrebate. '
                'Install with: pip install skrebate'
            ) from exc

        surf = MultiSURF(n_jobs=n_jobs)
        surf.fit(X, y)
        scores = np.asarray(getattr(surf, 'feature_importances_', None))

        if scores.size == 0:
            raise RuntimeError('MultiSURF did not provide feature_importances_')

        return scores

    raise ValueError(
        'Unsupported consensus_cv.selection_method. Expected one of: '
        '"mutual_info", "mutual_entropy", "multisurf", "none"'
    )

