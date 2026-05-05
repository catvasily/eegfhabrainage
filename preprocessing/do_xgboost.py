"""
**Run XGBoost classification on CWT amplitude distributions.**
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score
import xgboost
from xgboost import XGBClassifier
import sqlite3
import random
import pickle
import warnings

from tfdata import TFData
from plot_pr_curve import plot_and_save_pr_curve
from nested_cv import nested_cv
from consensus_cv import consensus_cv, build_and_fit_consensus_reducer, apply_consensus_reducer
from build_epi_features import build_and_fit_epi_features, apply_epi_features

STEP = 'xgboost'

def do_xgboost(ss):
    """
    Implements 'xgboost' step for the `run_classifier.py` script:
    performs XGBoost classification on CWT amplitude distributions.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    standardize = ss.args.get('standardize_features', False)
    ignore_confidence = ss.args.get('ignore_confidence', False)
    force_recalc = ss.args.get('force_recalc', False)
    cv_n_splits = ss.args.get('cv_n_splits', 5)
    hlist = ss.hlist(ss.args['hospital'])
    label = ss.args['target_label']
    nparms = 5 if ss.args.get('use_moments_only', False) else 9

    print(
        'Running XGBOOST step for: '
        f'hospital_list={hlist}, '
        f'label={label}, '
        f'nparms={nparms}, '
        f'standardize={standardize}, '
        f'ignore_confidence={ignore_confidence}'
    )

    pkl_pname = None
    if not force_recalc:
        pkl_pname = _find_existing_results_pickle(ss, standardize, ignore_confidence)

    # Note: using_precalc will always be false when force_recalc = True
    using_precalc = pkl_pname is not None and pkl_pname.exists()
    results_ready = False

    if using_precalc:
        print(f'Using precalculated results from: {pkl_pname}')

        with open(pkl_pname, 'rb') as fp:
            results_payload = pickle.load(fp)

        verify_pickle_xgboost_version(results_payload, pkl_pname)
        cv_n_splits = int(results_payload.get('cv_n_splits', cv_n_splits))
        Y = results_payload.get('Y')
        y_proba = results_payload.get('y_proba')
        y_score = results_payload.get('y_score')
        y_pred = results_payload.get('y_pred')
        results_ready = True
    else:
        Y, y_proba, y_score, y_pred, results_ready, cv_n_splits = _compute_classification_results(
            ss, standardize, cv_n_splits
        )

    if results_ready:
        # Calculate overall accuracy from predictions
        overall_accuracy = accuracy_score(Y, y_pred)
        print(f'CV accuracy: {overall_accuracy:.4f} (n_splits={cv_n_splits})')
        print('Classification report (CV predictions vs true labels):')
        print(classification_report(Y, y_pred))
        print('Confusion matrix:')
        print(confusion_matrix(Y, y_pred))

        if y_score is not None and np.unique(Y).size == 2:
            pr_auc = average_precision_score(Y, y_score)
            print(f'\nPR AUC (average precision): {pr_auc:.3f}\n')

        # Optionally compute and plot Precision-Recall curve (binary classification only)
        plot_pr = ss.args.get('plot_pr_curve', False)
        if plot_pr and y_score is not None and np.unique(Y).size == 2:
            show_plot = ss.args.get('show_plots', True)
            outfname = ss.prcrv_png_pname(
                hlist,
                ss.args['target_label'],
                nparms,
                standardize,
                ignore_confidence,
            )
            outfname = plot_and_save_pr_curve(
                Y,
                y_score,
                outfname=outfname,
                show_plot=show_plot,
                target_label=label,
                physician=ss.args.get('physician'),
            )
            print(f'Precision-Recall curve saved to: {outfname}')

    print(f'\n Step *{STEP}* completed successfully')


def _get_balanced_training_settings(config_dict):
    """
    Read balanced-training flags from config with backward compatibility.
    """
    enabled = bool(config_dict.get('balanced_training', False))
    ratio = float(config_dict.get('balance_ratio', 1.0))

    if ratio <= 0:
        raise ValueError(f'balance_ratio must be > 0, got {ratio}')

    return enabled, ratio

def _choose_balanced_subset_indices(y, balanced_ratio, seed=None):
    """
    Select sample indices by downsampling negatives to approx neg:pos ratio.

    Returns:
        selected(list of int): indecies of selected records
        info(dict): metadata with some useful info

    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(f'Expected 1D labels, got shape {y.shape}')

    neg_idx = np.flatnonzero(y == 0)
    pos_idx = np.flatnonzero(y == 1)

    n_neg = len(neg_idx)
    n_pos = len(pos_idx)

    if n_neg == 0 or n_pos == 0:
        return np.arange(len(y), dtype=int), {
            'applied': False,
            'reason': 'requires both negative and positive labels',
            'n_neg': n_neg,
            'n_pos': n_pos,
            'target_neg': None,
        }

    # Get the target number of negative  records
    target_neg = int(np.round(balanced_ratio * n_pos))
    target_neg = max(1, target_neg)

    if target_neg >= n_neg:
        # Can't achieve desired ratio
        return np.arange(len(y), dtype=int), {
            'applied': False,
            'reason': 'already at or below requested ratio',
            'n_neg': n_neg,
            'n_pos': n_pos,
            'target_neg': target_neg,
        }

    rng = np.random.default_rng(seed)

    # Randomly choose target_net indicies from all neg indicies
    # Note that original ordering of ned_idx is NOT preserved
    kept_neg = rng.choice(neg_idx, size=target_neg, replace=False)

    selected = np.concatenate([pos_idx, kept_neg])

    # Sort in ascending order so that relative ordering of records in
    # balanced and full datasets is the same
    selected = np.sort(selected.astype(int))

    return selected, {
        'applied': True,
        'reason': 'downsampled negatives',
        'n_neg': n_neg,
        'n_pos': n_pos,
        'target_neg': target_neg,
        'n_kept': int(selected.shape[0]),
    }


def _predict_with_full_model_data(raw_features, full_model_data):
    """
    Apply trained full model from payload to raw features and return predictions.
    """
    reducer = full_model_data['reducer']
    feature_names = full_model_data.get('feature_names', None)
    model = load_xgb_model_from_ubj_buffer(full_model_data['model_ubj'])

    X = apply_fitted_reducer(raw_features, reducer)

    if feature_names is not None and len(feature_names) == X.shape[1]:
        X_in = pd.DataFrame(X, columns=feature_names, copy=False)
    else:
        X_in = X

    y_proba = None
    y_score = None

    try:
        y_proba = model.predict_proba(X_in)
        y_pred = np.argmax(y_proba, axis=1)
        y_score = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
    except Exception:
        y_pred = model.predict(X_in)
        y_score = np.zeros(len(y_pred), dtype=float)

    return y_pred, y_score, y_proba


def _merge_balanced_and_discarded_outputs(
    n_total,
    balanced_idx,           # Mapping of balanced subset into full dataset
    discarded_idx,          # Mapping of discarded subset into full dataset
    y_pred_balanced,        # ..._balanced: predictions on the balanced part
    y_score_balanced,
    y_proba_balanced,
    y_pred_discarded,       # ..._discarded: predictions on the discarded part
    y_score_discarded,
    y_proba_discarded,
):
    """
    Merge prediction outputs from balanced CV subset and discarded subset.

    Returns:
        y_pred, y_score, y_proba: predictions on the full data

    """
    balanced_idx = np.asarray(balanced_idx, dtype=int)
    discarded_idx = np.asarray(discarded_idx, dtype=int)

    y_pred_all = np.empty(n_total, dtype=np.asarray(y_pred_balanced).dtype)
    y_pred_all[balanced_idx] = np.asarray(y_pred_balanced)
    y_pred_all[discarded_idx] = np.asarray(y_pred_discarded)

    y_score_all = None
    if y_score_balanced is not None or y_score_discarded is not None:
        y_score_all = np.full(n_total, np.nan, dtype=float)
        if y_score_balanced is not None:
            y_score_all[balanced_idx] = np.asarray(y_score_balanced, dtype=float)
        if y_score_discarded is not None:
            y_score_all[discarded_idx] = np.asarray(y_score_discarded, dtype=float)

    y_proba_all = None
    n_classes = None
    if y_proba_balanced is not None:
        n_classes = y_proba_balanced.shape[1]
    elif y_proba_discarded is not None:
        n_classes = y_proba_discarded.shape[1]

    if n_classes is not None:
        y_proba_all = np.full((n_total, n_classes), np.nan, dtype=float)
        if y_proba_balanced is not None and y_proba_balanced.shape[1] == n_classes:
            y_proba_all[balanced_idx, :] = np.asarray(y_proba_balanced, dtype=float)
        if y_proba_discarded is not None and y_proba_discarded.shape[1] == n_classes:
            y_proba_all[discarded_idx, :] = np.asarray(y_proba_discarded, dtype=float)

    return y_pred_all, y_score_all, y_proba_all

def _compute_classification_results(ss, standardize, cv_n_splits):
    """
    Compute classification results by loading data, training model, and running cross-validation.
    Depending on JSON settings, this may be a standard cross-validation or nested
    cross-validation with parameters tuning. The data may be optionally balanced.
    
    This function runs only when results are not ready, or when ready results
    exist but recalculation was explicitly requested.

    Args:
        ss(obj): application object
        standardize(bool): whether to standardize features
        cv_n_splits(int): number of cross-validation folds
    
    Returns:
        tuple: (Y, y_proba, y_score, y_pred, results_ready, cv_n_splits_used)

    """
    # --------------------------------------
    # Load data and labels and initial setup
    # --------------------------------------
    lst_IDs = ss.args['scan_ids']

    if lst_IDs is None:
        # Get list of pairs (shospital #, scan_id) for all requested
        # hospitals
        lst_IDs = list_src_ids(ss)
        print(f'# of EEG records: {len(lst_IDs)}')
    else:
        if len(ss.args['hospital']) > 1:
            raise ValueError('When scan_ids are not null, only one hospital should be listed')

        # Reformat the list to a list of tuples (id, 0)
        lst_IDs = [(0,id) for id in lst_IDs]

    # Now ss.args['scan_ids'] is actually a list of tuples (id, idx_hospital)
    ss.args['scan_ids'] = lst_IDs

    # Load labels for scan IDs in lst_IDs. Labels for some of them may be
    # missing.
    Y = load_labels(ss)    # Y = vector of nscans labels
                           # (ss.args['scan_ids'] updated)

    # Load raw features (not flattened)
    raw_features, ch_names, tfd_freqs, tfd_parm_names = load_data(ss)  # (nscans, nchans, nfreqs, nparms)

    # Keep original unbalanced labels/features for final metrics recalculation.
    Y_unbalanced = np.asarray(Y)
    raw_features_unbalanced = raw_features
    balanced_idx = np.arange(len(Y_unbalanced), dtype=int)
    discarded_idx = np.array([], dtype=int)
    balance_applied = False

    balanced_training_requested, balanced_ratio = _get_balanced_training_settings(ss.args)
    balanced_training_active = balanced_training_requested  # Renaming is for future modifications
                                                            # where 'requested' and 'active' may not always
                                                            # be the same

    if balanced_training_active:
        seed = ss.args.get('seed', None)

        # Balanced idx are the indecies of selected records, info contains some metadata
        # like how many neg labels were actually kept
        balanced_idx, balance_info = _choose_balanced_subset_indices(Y, balanced_ratio, seed=seed)

        if balance_info.get('applied', False):
            balance_applied = True
            discarded_idx = np.setdiff1d(np.arange(len(Y_unbalanced), dtype=int), balanced_idx)

            # if 'applied' is true - then balancing actually happened (for indecies)
            raw_features = raw_features[balanced_idx]   # Now just apply it
            Y = np.asarray(Y)[balanced_idx]
            ss.args['scan_ids'] = [ss.args['scan_ids'][i] for i in balanced_idx]
            print(
                'Balanced training active: '
                f"kept {len(balanced_idx)} / {len(Y_unbalanced)} records "
                f"(n_pos={balance_info['n_pos']}, n_neg kept={balance_info['target_neg']} of {balance_info['n_neg']})"
            )
        else:
            print(f"Balanced training requested but no downsampling applied ({balance_info.get('reason', 'n/a')})")

    # Remove FS-added channels with no clear lobe assignments
    drop_channels = {'Unknown-lh', 'Unknown-rh'}
    keep_idx = [i for i, name in enumerate(ch_names) if name not in drop_channels]
    raw_features = raw_features[:, keep_idx, :, :]
    raw_features_unbalanced = raw_features_unbalanced[:, keep_idx, :, :]
    ch_names = [ch_names[i] for i in keep_idx]

    # Pass channel names, etc for reducers that need atlas mappings (e.g., to_lobes)
    ss.args['ch_names'] = ch_names
    ss.args['freqs'] = tfd_freqs
    ss.args['parm_names'] = tfd_parm_names

    assert raw_features.shape[0] == len(Y)
    print(f'Raw feature shape: {raw_features.shape}')

    # Seed random generators if requested in config
    seed = ss.args.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            ss.rng = np.random.default_rng(seed=seed)
        except Exception:
            pass

    # XGBoost classifier parameters from config
    xgb_params = ss.args.get('xgb_params', {}) or {}

    # initialize random gens there too
    if seed is not None and 'random_state' not in xgb_params:
        xgb_params['random_state'] = seed

    # Cross-validation setup
    # Method split(X,y) of the StratifiedKFold object returns iterator
    # of cv_n_spits tuples (train_indices, validation_indices)
    kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=seed)

    # Initialize classifier
    try:
        clf = XGBClassifier(**xgb_params)
    except Exception as e:
        raise RuntimeError('Failed to initialize XGBClassifier; ensure xgboost is installed and params are valid') from e

    # Evaluate with cross-validation if there are at least 2 label classes
    unique_labels = np.unique(Y)

    results_ready = False
    nested_cv_info = None
    full_model_clf = clf
    y_proba = None  # Class probability matrix, shape (nsamples, nclasses). For each sample,
                    # these are class probabilities estimated when it was in a test fold
    y_score = None  # 1D score used for PR curve logic: the second probability column (y_proba[:, 1]) when available,
                    # otherwise flattened single-column probs, shape (n_samples,) 
    y_pred = None   # predicted class from argmax(y_proba, axis=1); corresponds to PR curve 50%
                    # threshold for binary classifications

    if unique_labels.size < 2:
        print('Not enough label classes for training (need at least 2). Skipping classification.')
        return Y, y_proba, y_score, y_pred, results_ready, cv_n_splits

    use_nested_grid_search = bool(ss.args.get('use_nested_grid_search', False))
    use_consensus_cv = bool(ss.args.get('use_consensus_cv', False))
    consensus_cfg = ss.args.get('consensus_cv', {}) or {}
    nested_cfg = ss.args.get('nested_cv', {}) or {}

    # Check if per-fold dimensionality reduction is requested
    per_fold_reduction = ss.args.get('per_fold_dim_reduction', False)

    if standardize and not use_consensus_cv:
        # To prevent data leakage, one cannot standardize on full data.
        # This is why we'll need per_fold_reduction turned on
        per_fold_reduction = True

    if use_consensus_cv and use_nested_grid_search:
        raise ValueError(
            'use_consensus_cv and use_nested_grid_search cannot be enabled at the same time. '
            'Use consensus_cv.grid_tuning for optional tuning with consensus CV.'
        )

    if use_consensus_cv and per_fold_reduction:
        raise ValueError(
            'use_consensus_cv already performs per-outer-fold feature selection/reduction; '
            'set per_fold_dim_reduction=false.'
        )

    # ----- end of all kinds of setups and initializations -----------------

    # -------------------
    # Consensus CV branch
    # -------------------
    if use_consensus_cv:
    # -------------------
        outer_n_splits = int(consensus_cfg.get('outer_n_splits', cv_n_splits))
        inner_n_splits = int(consensus_cfg.get('inner_n_splits', 5))

        if outer_n_splits < 2 or inner_n_splits < 2:
            raise ValueError('consensus_cv outer_n_splits and inner_n_splits must be >= 2')

        cv_n_splits = outer_n_splits
        outer_kf = StratifiedKFold(n_splits=outer_n_splits, shuffle=True, random_state=seed)

        selection_method = str(consensus_cfg.get('selection_method', 'mutual_info')).lower()

        if selection_method in ('mutual_entropy', 'mutual_entrophy'):
            selection_method = 'mutual_info'

        print(
            f'Running consensus CV '
            f'(outer={outer_n_splits}, inner={inner_n_splits}, selection={selection_method})...'
        )

        scores, y_pred, y_score, y_proba, fold_summaries = consensus_cv(
            estimator=clf,
            raw_features=raw_features,
            y=Y,
            outer_cv_splitter=outer_kf,
            consensus_cfg=consensus_cfg,
            standardize=standardize,
            seed=seed,
        )

        selected_consensus_params = _select_best_nested_params(fold_summaries)

        if selected_consensus_params is not None:
            full_model_clf = clone(clf)
            full_model_clf.set_params(**selected_consensus_params)
            print(f'Using consensus-CV selected params for full-data model fit: {selected_consensus_params}')

        nested_cv_info = {
            'enabled': False,
        }

        consensus_cv_info = {
            'enabled': True,
            'outer_n_splits': outer_n_splits,
            'inner_n_splits': inner_n_splits,
            'selection_method': selection_method,
            'grid_tuning': consensus_cfg.get('grid_tuning', {}) or {},
            'outer_fold_scores': np.asarray(scores),
            'fold_summaries': fold_summaries,
            'selected_full_model_params': selected_consensus_params,
            'selected_full_model_params_metric': 'best_inner_score',
        }

    # ---------------------------------
    # Nested CV parameter tuning branch
    # ---------------------------------
    elif use_nested_grid_search:
    # ---------------------------------
        if per_fold_reduction:
            raise ValueError(
                'use_nested_grid_search and per_fold_dim_reduction cannot be used together. '
                'Set per_fold_dim_reduction=false when nested tuning is enabled.'
            )

        param_grid = nested_cfg.get('param_grid', None)

        if not isinstance(param_grid, (dict, list)) or len(param_grid) == 0:
            raise ValueError(
                'use_nested_grid_search is enabled, but nested_cv.param_grid is missing or empty'
            )

        # Typically one wants to set outer and inner_n_splits ~5; do this in JSON
        outer_n_splits = int(nested_cfg.get('outer_n_splits', cv_n_splits))
        inner_n_splits = int(nested_cfg.get('inner_n_splits', 3))

        if outer_n_splits < 2 or inner_n_splits < 2:
            raise ValueError('nested_cv outer_n_splits and inner_n_splits must be >= 2')

        cv_n_splits = outer_n_splits    # cv_n_splits is the n_splits setting for ordinary
                                        # CV without tuning - for consistency

        # These splitter objects, when called with split(X,Y) method, return an iterator
        # (train_idx, test_idx) for splitting data and labels into training and test subsets
        outer_kf = StratifiedKFold(n_splits=outer_n_splits, shuffle=True, random_state=seed)
        inner_kf = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)

        X, _ = reduce_dimensions(raw_features, ss.args, standardize=standardize)
        print(f'Data standardization: {"ON" if standardize else "OFF"}')
        print(f'Final feature vector dimensions: {X.shape[1]}')

        dim_reduction_method = str(ss.args.get('dim_reduction', 'pca')).lower()

        if standardize or dim_reduction_method == 'pca':
            print(
                'WARNING: nested GridSearchCV is running on a precomputed X matrix. '
                'For strict leakage control with PCA/standardization, move these steps '
                'inside the nested outer-fold loop.'
            )

        scoring = nested_cfg.get('scoring', 'accuracy')
        grid_n_jobs = int(nested_cfg.get('n_jobs', 1))  # set to -1 to use all available
                                                        # CPUs, set to -2 to use all but
                                                        # one CPU (computer will be more responsive)
        grid_verbose = int(nested_cfg.get('verbose', 0))

        print(
            f'Running nested CV with GridSearchCV '
            f'(outer={outer_n_splits}, inner={inner_n_splits}, scoring={scoring})...'
        )
        scores, y_pred, y_score, y_proba, fold_summaries = nested_cv(
            estimator=clf,
            X=X,
            Y=Y,
            outer_cv_splitter=outer_kf,
            inner_cv_splitter=inner_kf,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=grid_n_jobs,
            verbose=grid_verbose,
        )

        # The returned fold_summaries is a list of dicts with keys:
        #    'fold': ifold,
        #    'n_train': len(train_idx),
        #    'n_test': len(test_idx),
        #    'best_params': dict(grid.best_params_),
        #    'best_inner_score': grid.best_score_,
        #    'outer_test_accuracy': fold_acc

        best_nested_params = _select_best_nested_params(fold_summaries)

        if best_nested_params is not None:
            full_model_clf = clone(clf)
            full_model_clf.set_params(**best_nested_params)
            print(f'Using nested-CV best params for full-data model fit: {best_nested_params}')

        nested_cv_info = {
            'enabled': True,
            'outer_n_splits': outer_n_splits,
            'inner_n_splits': inner_n_splits,
            'scoring': scoring,
            'n_jobs': grid_n_jobs,
            'verbose': grid_verbose,
            'param_grid': param_grid,
            'outer_fold_scores': np.asarray(scores),
            'fold_summaries': fold_summaries,
            'selected_full_model_params': best_nested_params,
            'selected_full_model_params_metric': 'best_inner_score',
        }
        consensus_cv_info = {'enabled': False}

    # ---------------------------------------
    # Standard CV with per-fold dim reduction
    # ---------------------------------------
    elif per_fold_reduction:
    # ---------------------------------------
        # Use this for PCA dim reduction to prevent data leakage 
        print(f"Running stratified {cv_n_splits}-fold CV with PER-FOLD dimensionality reduction...")
        scores, y_pred, y_score, y_proba = cross_validate_with_per_fold_reduction(
            clf, raw_features, Y, kf, ss.args, standardize
        )
        consensus_cv_info = {'enabled': False}
        nested_cv_info = {'enabled': False}

    # -------------------------------------
    # Standard CV with global dim reduction
    # -------------------------------------
    else:
        # Standard approach: reduce dimensions on full dataset, then cross-validate
        X, _ = reduce_dimensions(raw_features, ss.args, standardize=standardize)
        print(f'Data standardization: {"ON" if standardize else "OFF"}')
        print(f'Final feature vector dimensions: {X.shape[1]}')
        print(f"Running stratified {cv_n_splits}-fold CV using XGBoost...")

        # Get cross-validated probabilities, derive predictions and scores
        # With method='predict_proba', cross_val_predict() returns
        # y_proba = 2D array of class probabilities, shape (n_samples, n_classes)
        # of probabilities of each class for this sample
        y_proba = cross_val_predict(clf, X, Y, cv=kf, method='predict_proba', n_jobs=-1)

        # Derive class predictions from probabilities (argmax)
        y_pred = np.argmax(y_proba, axis=1)
        # If binary, take probability of positive class for PR curve
        # Otherwise - the first non-zero label
        if y_proba.shape[1] > 1:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba.ravel()
        consensus_cv_info = {'enabled': False}
        nested_cv_info = {'enabled': False}

    # Done with all CVs and performance estimations

    # ------------------------------------------------------------
    # Train on all available data, and return everything required
    # for post_hoc analyses
    # ------------------------------------------------------------
    nparms = raw_features.shape[3]

    # full_model_data is a dict with trained model, fitted reducer, feature names,
    # labels, scan IDs, and metadata needed to recompute reduced features later.
    if use_consensus_cv:
        full_model_data = _construct_full_model_data_consensus_cv(
            clf=full_model_clf,
            raw_features=raw_features,
            Y=Y,
            fold_summaries=fold_summaries,
            config_dict=ss.args,
            standardize=standardize,
        )
    else:
        full_model_data = _construct_full_model_data(
            clf=full_model_clf,
            raw_features=raw_features,      # Note that this is still balanced data, if balancing is on
            Y=Y,
            config_dict=ss.args,
            standardize=standardize,
        )

    if balance_applied and len(discarded_idx) > 0:
        # Keep balanced CV outputs and only infer discarded negatives with
        # the full classifier trained on balanced data.
        y_pred_discarded, y_score_discarded, y_proba_discarded = _predict_with_full_model_data(
            raw_features_unbalanced[discarded_idx], full_model_data
        )

        y_pred, y_score, y_proba = _merge_balanced_and_discarded_outputs(
            n_total=len(Y_unbalanced),
            balanced_idx=balanced_idx,
            discarded_idx=discarded_idx,
            y_pred_balanced=y_pred,
            y_score_balanced=y_score,
            y_proba_balanced=y_proba,
            y_pred_discarded=y_pred_discarded,
            y_score_discarded=y_score_discarded,
            y_proba_discarded=y_proba_discarded,
        )

        Y = Y_unbalanced
        print(
            'Performance outputs merged to original unbalanced data: '
            f'CV predictions on balanced subset + full-model predictions on {len(discarded_idx)} discarded records'
        )

    _save_classification_results(
        ss,
        nparms,
        Y,
        y_proba,
        y_score,
        y_pred,
        seed,
        cv_n_splits,
        standardize,
        full_model_data,
        nested_cv_info=nested_cv_info,
        consensus_cv_info=consensus_cv_info,
    )

    results_ready = True

    return Y, y_proba, y_score, y_pred, results_ready, cv_n_splits

def _find_existing_results_pickle(ss, standardize, ignore_confidence):
    hlist = ss.hlist(ss.args['hospital'])
    label = ss.args['target_label']
    nparms = 5 if ss.args.get('use_moments_only', False) else 9
    exact = ss.cls_pkl_pname(hlist, label, nparms, standardize, ignore_confidence)
    if exact.exists():
        return exact

    return None

def _select_best_nested_params(fold_summaries):
    """
    Choose a single hyperparameter set from nested-CV fold summaries.
    Current selection rule: params from fold with highest best_inner_score.

    """
    if not isinstance(fold_summaries, list) or len(fold_summaries) == 0:
        return None

    best_summary = None
    best_score = None

    for summary in fold_summaries:
        if not isinstance(summary, dict):
            continue

        score = summary.get('best_inner_score', None)
        params = summary.get('best_params', None)

        if score is None or not isinstance(params, dict):
            continue

        if best_score is None or float(score) > best_score:
            best_score = float(score)
            best_summary = summary

    if best_summary is None:
        return None

    return dict(best_summary.get('best_params', {}))

def _save_classification_results(
    ss,
    nparms,
    Y,
    y_proba,
    y_score,
    y_pred,
    seed,
    cv_n_splits,
    standardize,
    full_model_data,
    nested_cv_info=None,
    consensus_cv_info=None,
):
    use_moments_only = ss.args.get('use_moments_only', False)
    ignore_confidence = ss.args.get('ignore_confidence', False)
    hlist = ss.hlist(ss.args['hospital'])
    pkl_pname = ss.cls_pkl_pname(hlist,ss.args['target_label'], nparms, standardize, ignore_confidence)
    pkl_pname.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        'Y': Y,
        'y_proba': y_proba,
        'y_score': y_score,
        'y_pred': y_pred,
        'xgboost_version': _runtime_xgboost_version(),
        'hospital': ss.args['hospital'],
        'physician': ss.args.get('physician'),
        'target_label': ss.args['target_label'],
        'seed': seed,
        'cv_n_splits': cv_n_splits,
        'use_moments_only': use_moments_only,
        'standardize': standardize,
        'ignore_confidence': ignore_confidence,
        'full_model_data': full_model_data,
        'nested_cv': nested_cv_info,
        'consensus_cv': consensus_cv_info,
    }

    if pkl_pname.exists():
        pkl_pname = pkl_pname.with_name(f'new_{pkl_pname.name}')
        print(f'Warning: pickle exists, saving to new file: {pkl_pname}')

    with open(pkl_pname, 'wb') as fp:
        pickle.dump(results_payload, fp)

    print(f'Saved classification results to: {pkl_pname}')

def _construct_full_model_data(clf, raw_features, Y, config_dict, standardize):
    """
    Build and fit a final model on all available samples and return all items 
    required for post-hoc feature-importance and SHAP analysis.

    Returns:
        dict with trained model, fitted reducer, feature names, labels,
        scan IDs, and metadata needed to recompute reduced features later.
    """
    X_reduced, reducer = reduce_dimensions(raw_features, config_dict, standardize=standardize)
    feature_names = _build_feature_names(X_reduced.shape[1], config_dict, reducer)

    X_df = pd.DataFrame(X_reduced, columns=feature_names, copy=False)   # Request avoiding duplicating data
    final_clf = clone(clf)
    print(f'Training full model on {X_df.shape[0]} samples, {X_df.shape[1]} features...')
    final_clf.fit(X_df, Y)
    print('Full model training done.')

    model_ubj = serialize_xgb_model_to_ubj_buffer(final_clf)

    return {
        'model_ubj': model_ubj,
        'reducer': reducer,
        'feature_names': feature_names,
        'labels': np.asarray(Y),
        'scan_ids': list(config_dict.get('scan_ids', [])),
        'dim_reduction': config_dict.get('dim_reduction', 'pca'),
        'standardize': standardize,
        'requires_feature_recompute': True,
        'feature_recompute_note': 'Reduced feature matrix is intentionally omitted to save disk space; rerun dimensionality reduction after loading if features are needed for SHAP/background data.',
    }


def _construct_full_model_data_consensus_cv(
    clf,
    raw_features,
    Y,
    fold_summaries,
    config_dict,
    standardize,
):
    """
    Fit full-data model for consensus CV mode and return reusable payload.

    Reuses the feature set from the best-performing outer fold (by
    outer_test_accuracy) instead of running a new feature selection pass on the
    full dataset.
    """
    best_fold = max(fold_summaries, key=lambda fs: fs['outer_test_accuracy'])
    selected_idx = np.asarray(best_fold['selected_idx'], dtype=int)
    n_features_in = int(best_fold['n_features_in'])

    print(
        f'Full-model feature set: reusing outer fold {best_fold["fold"]} '
        f'(accuracy={best_fold["outer_test_accuracy"]:.4f}, '
        f'{len(selected_idx)} features selected)'
    )

    X_flat = raw_features.reshape(raw_features.shape[0], -1)
    if X_flat.shape[1] != n_features_in:
        raise ValueError(
            f'Full-data feature dimension mismatch: fold reducer expects '
            f'{n_features_in} flat features, got {X_flat.shape[1]}'
        )
    X_selected = X_flat[:, selected_idx]

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
        'n_features_in': n_features_in,
        'selection_method': best_fold.get('selection_method', ''),
    }

    feature_names = _build_consensus_feature_names(selected_idx, config_dict)

    X_df = pd.DataFrame(X_reduced, columns=feature_names, copy=False)
    final_clf = clone(clf)
    print(f'Training full model on {X_df.shape[0]} samples, {X_df.shape[1]} features...')
    final_clf.fit(X_df, Y)
    print('Full model training done.')

    model_ubj = serialize_xgb_model_to_ubj_buffer(final_clf)

    return {
        'model_ubj': model_ubj,
        'reducer': reducer,
        'feature_names': feature_names,
        'labels': np.asarray(Y),
        'scan_ids': list(config_dict.get('scan_ids', [])),
        'dim_reduction': 'consensus_cv',
        'standardize': standardize,
        'requires_feature_recompute': True,
        'feature_recompute_note': 'Consensus-selected reduced feature matrix is intentionally omitted to save disk space; rerun reduction after loading if features are needed for SHAP/background data.',
    }

def _resolve_atlas_csv_path(config_dict):
    """
    Resolve atlas CSV path from config with the same fallback used by to_lobes.
    """
    to_lobes_config = config_dict.get('to_lobes', {}) or {}
    atlas_csv = to_lobes_config.get('atlas_csv')

    if atlas_csv is None:
        atlas_csv = Path(__file__).with_name('destrieux_atlas_ordered.csv')
    else:
        atlas_csv = Path(atlas_csv)
        if not atlas_csv.is_absolute():
            atlas_csv = Path(__file__).parent / atlas_csv

    if not atlas_csv.exists():
        raise FileNotFoundError(f'Atlas CSV not found: {atlas_csv}')

    return atlas_csv

def _build_consensus_feature_names(selected_idx, config_dict):
    """
    Build consensus feature names from selected flat indices.

    Name format:
        <group>|ch<channel_idx>|<frequency>.2fHz|<parameter>
    """
    ch_names = config_dict.get('ch_names', [])
    freqs = np.asarray(config_dict.get('freqs', []), dtype=float)
    parm_names = list(config_dict.get('parm_names', []))

    if not ch_names:
        raise ValueError('consensus_cv feature naming requires config_dict["ch_names"]')
    if freqs.size == 0:
        raise ValueError('consensus_cv feature naming requires config_dict["freqs"]')
    if not parm_names:
        raise ValueError('consensus_cv feature naming requires config_dict["parm_names"]')

    atlas_csv = _resolve_atlas_csv_path(config_dict)
    atlas_df = _load_destrieux_atlas(atlas_csv)
    group_indices, group_labels = _build_lobe_groups(ch_names, atlas_df)

    ch_to_group = {}
    for grp, idxs in zip(group_labels, group_indices):
        for ch_idx in idxs:
            ch_to_group[int(ch_idx)] = grp

    nchans = len(ch_names)
    nfreqs = freqs.shape[0]
    nparms = len(parm_names)
    nflat_total = nchans * nfreqs * nparms

    names = []
    for flat_idx in np.asarray(selected_idx, dtype=int):
        if flat_idx < 0 or flat_idx >= nflat_total:
            raise ValueError(
                f'consensus_cv selected_idx out of bounds: {flat_idx} not in [0, {nflat_total})'
            )

        ch_idx, ifreq, iparm = np.unravel_index(int(flat_idx), (nchans, nfreqs, nparms))
        grp = ch_to_group.get(int(ch_idx), 'UNK')
        frq = float(freqs[ifreq])
        parm = parm_names[iparm]
        names.append(f'{grp}|ch{ch_idx}|{frq:.2f}Hz|{parm}')

    return names

def _build_feature_names(n_features, config_dict, reducer):
    method = reducer.get('method', config_dict.get('dim_reduction', 'pca'))

    if method == 'pca':
        return [f'P{i}' for i in range(n_features)]

    freqs = config_dict.get('freqs', [])
    parm_names = config_dict.get('parm_names', [])

    if method == 'none':
        ch_names = config_dict.get('ch_names', [])
        names = [
                f'{ch}|{frq:.2f}Hz|{parm}'
            for ch in ch_names
            for frq in freqs
            for parm in parm_names
        ]
        if len(names) == n_features:
            return names

    if method == 'epi_features':
        epi_parm_names = reducer.get('epi_parm_names', [])
        nchans = len(config_dict.get('ch_names', []))
        names = [
            f'c{ich}|{parm}'
            for ich in range(nchans)
            for parm in epi_parm_names
        ]
        if len(names) == n_features:
            return names

    if method == 'to_lobes':
        group_labels = reducer.get('group_labels', [])
        used_freqs = reducer.get('freqs', freqs)
        names = [
            f'{grp}|{frq:.2f}Hz|{parm}'
            for grp in group_labels
            for frq in used_freqs
            for parm in parm_names
        ]
        if len(names) == n_features:
            return names

    # This is a fall-back generic option for any dim reductin method
    # not listed above:
    return [f'f{i}' for i in range(n_features)]

def cross_validate_with_per_fold_reduction(clf, raw_features, Y, cv_splitter, config_dict, standardize=False):
    """
    Perform cross-validation with dimensionality reduction applied separately for each fold.
    
    For each fold: fit dimensionality reduction on training set only, then apply the fitted
    transformer to test set. This prevents information leakage from test sets.

    Args:
        clf(XGBClassifier): initialized classifier
        raw_features(ndarray): `shape (nscans, nchans, nfreqs, nparms)` - raw 3D features
        Y(ndarray): labels vector
        cv_splitter(StratifiedKFold): cross-validation splitter
        config_dict(dict): configuration dictionary with dimensionality reduction params
        standardize(bool): whether to standardize features before reduction

    Returns:
        scores(ndarray): cross-validation fold scores
        y_pred(ndarray): cross-validated predictions for all samples
        y_score(ndarray): cross-validated predicted probability (score) for the
            positive class when available (same length as `Y`). If probabilities
            cannot be computed for any reason, contains zeros.
        y_proba(ndarray|None): cross-validated predicted probabilities for all
            classes when available, shape (n_samples, n_classes). If probabilities
            cannot be computed for any reason, returns None.
    """
    nscans = raw_features.shape[0]
    y_pred_all = np.zeros_like(Y, dtype=float)
    # Store predicted probability (score) for positive class when available
    y_score_all = np.zeros_like(Y, dtype=float)
    scores_list = []
    y_proba_all = None

    for ifold, (train_idx, test_idx) in enumerate(cv_splitter.split(raw_features, Y)):
        print(f'  Fold {ifold + 1}/{cv_splitter.get_n_splits()}')
        
        # Extract training and test raw features
        raw_train = raw_features[train_idx]
        raw_test = raw_features[test_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        # Fit dimensionality reduction on training data and get the fitted transformer
        X_train, transformer = reduce_dimensions(raw_train, config_dict, standardize=standardize)
        
        # Apply the fitted transformer to test data (no fitting on test data)
        X_test = apply_fitted_reducer(raw_test, transformer)

        # Train classifier on this fold
        clf_fold = XGBClassifier(**clf.get_params())
        clf_fold.fit(X_train, Y_train)

        # Evaluate on test set
        # Prefer probabilities, then derive predictions to avoid duplicate passes
        try:
            y_proba_fold = clf_fold.predict_proba(X_test)
            y_pred_fold = np.argmax(y_proba_fold, axis=1)
            if y_proba_fold.shape[1] > 1:
                y_score_fold = y_proba_fold[:, 1]
            else:
                y_score_fold = y_proba_fold.ravel()
        except Exception:
            y_pred_fold = clf_fold.predict(X_test)
            y_score_fold = np.zeros(len(y_pred_fold))
            y_proba_fold = None

        fold_score = accuracy_score(Y_test, y_pred_fold)
        scores_list.append(fold_score)

        # Store predictions and scores in correct positions
        y_pred_all[test_idx] = y_pred_fold
        y_score_all[test_idx] = y_score_fold
        if y_proba_fold is not None:
            if y_proba_all is None:
                y_proba_all = np.zeros((nscans, y_proba_fold.shape[1]), dtype=float)
            y_proba_all[test_idx, :] = y_proba_fold

        print(f'    Fold accuracy: {fold_score:.4f}')

    scores = np.array(scores_list)
    return scores, y_pred_all, y_score_all, y_proba_all

def reduce_dimensions(raw_features, config_dict, standardize=False):
    """
    Build and fit a dimensionality reduction transformer on raw features.

    This function fits the transformer on the provided data and returns both
    the transformed data and the fitted transformer object for later use on
    different data (e.g., test set).

    The method dispatches to specific reducer implementations based on 'dim_reduction'
    config key. Each reducer method must:

      1. Accept (raw_features, config_dict, standardize) as arguments
      2. Build an sklearn pipeline (possibly with StandardScaler + dimensionality reduction step)
      3. Fit the pipeline on the input data
      4. Return tuple of (transformed_features, transformer_dict)
      5. transformer_dict must contain 'method' (str) and 'pipeline' (fitted Pipeline object) keys

    Supported reducer methods:

    - 'pca': Principal Component Analysis via _build_and_fit_pca().
    - 'none': No reduction, optional standardization only via _build_and_fit_none().
    - 'to_lobes': Reduce channels to hemisphere/lobe groups via _build_and_fit_to_lobes().
    - 'epi_features': EEG-derived per-channel features via build_epi_features.py.

    Args:
        raw_features(ndarray): `shape (nscans, nchans, nfreqs, nparms)` - raw 3D features
        config_dict(dict): configuration dictionary with 'dim_reduction' key specifying method,
                           plus method-specific config (e.g., 'pca' key for PCA parameters)
        standardize(bool): whether to standardize features before reduction (prepended to pipeline)

    Returns:
        X(ndarray): `shape (nscans, feature_dim)` - transformed features
        transformer(dict): fitted transformer dictionary containing:

            - 'method': name of the method used (str)
            - 'pipeline': fitted sklearn Pipeline object (can apply via apply_fitted_reducer)
    """
    method = config_dict.get('dim_reduction', 'pca')
    print(f'Dimensionality reduction method: {method}')

    if method == 'pca':
        nscans = raw_features.shape[0]
        # Flatten: (nscans, nchans, nfreqs, nparms) -> (nscans, nchans*nfreqs*nparms)
        features_flat = raw_features.reshape(nscans, -1)
        X, transformer = _build_and_fit_pca(features_flat, config_dict, standardize)
    
    elif method == 'none':
        nscans = raw_features.shape[0]
        # Flatten: (nscans, nchans, nfreqs, nparms) -> (nscans, nchans*nfreqs*nparms)
        features_flat = raw_features.reshape(nscans, -1)
        X, transformer = _build_and_fit_none(features_flat, config_dict, standardize)

    elif method == 'to_lobes':
        X, transformer = _build_and_fit_to_lobes(raw_features, config_dict, standardize)

    elif method == 'epi_features':
        X, transformer = build_and_fit_epi_features(raw_features, config_dict, standardize)
    
    else:
        raise ValueError(f'Unknown dimensionality reduction method: {method}')

    return X, transformer

def apply_fitted_reducer(raw_features, transformer):
    """
    Apply a previously fitted dimensionality reduction transformer to new raw features.
    
    Args:
        raw_features(ndarray): `shape (nscans, nchans, nfreqs, nparms)` - raw 3D features
        transformer(dict): fitted transformer dictionary from reduce_dimensions()

    Returns:
        X(ndarray): `shape (nscans, feature_dim)` - transformed features using the fitted transformer
    """
    method = transformer['method']
    pipeline = transformer['pipeline']

    if method == 'pca':
        nscans = raw_features.shape[0]
        # Flatten: (nscans, nchans, nfreqs, nparms) -> (nscans, nchans*nfreqs*nparms)
        features_flat = raw_features.reshape(nscans, -1)
        X = pipeline.transform(features_flat)
    
    elif method == 'none':
        nscans = raw_features.shape[0]
        # Flatten: (nscans, nchans, nfreqs, nparms) -> (nscans, nchans*nfreqs*nparms)
        features_flat = raw_features.reshape(nscans, -1)
        X = pipeline.transform(features_flat)

    elif method == 'to_lobes':
        X = _apply_to_lobes(raw_features, transformer)

    elif method == 'epi_features':
        X = apply_epi_features(raw_features, transformer)

    elif method == 'consensus_cv':
        X = apply_consensus_reducer(raw_features, transformer)
    
    else:
        raise ValueError(f'Unknown dimensionality reduction method: {method}')

    return X

def _build_and_fit_pca(features_flat, config_dict, standardize=False):
    """
    Build and fit PCA-based dimensionality reduction.

    Args:
        features_flat(ndarray): `shape (nscans, nfeatures)` - flattened features
        config_dict(dict): configuration dictionary with 'pca' key containing method params
        standardize(bool): whether to standardize features before PCA

    Returns:
        X(ndarray): `shape (nscans, n_components)` - PCA-reduced features
        transformer(dict): fitted transformer dictionary with 'method' and 'pipeline' keys
    """
    # Build pipeline
    pipeline_steps = []
    if standardize:
        pipeline_steps.append(('scaler', StandardScaler()))

    pca_config = config_dict.get('pca', {})
    n_components = pca_config.get('n_components', 0.95)
    
    if isinstance(n_components, int):
        print(f'  PCA: reducing to fixed number of {n_components} components')
    elif isinstance(n_components, float):
        assert (n_components > 0) and (n_components <= 1)
        print(f'  PCA: reducing to {n_components*100}% of variance')
    else:
        raise ValueError(f'Invalid type for PCA n_components: {n_components}')
    
    pipeline_steps.append(('pca', PCA(n_components=n_components)))

    pipeline = make_pipeline(*[step for _, step in pipeline_steps])
    X = pipeline.fit_transform(features_flat)

    transformer = {
        'method': 'pca',
        'pipeline': pipeline
    }
    return X, transformer

def _build_and_fit_none(features_flat, config_dict, standardize=False):
    """
    Build and fit 'none' dimensionality reduction (just flatten and optionally standardize).

    Args:
        features_flat(ndarray): `shape (nscans, nfeatures)` - flattened features
        config_dict(dict): configuration dictionary (unused for this method)
        standardize(bool): whether to standardize features

    Returns:
        X(ndarray): `shape (nscans, nfeatures)` - flattened (and optionally standardized) features
        transformer(dict): fitted transformer dictionary with 'method' and 'pipeline' keys

    """
    pipeline_steps = []
    if standardize:
        pipeline_steps.append(('scaler', StandardScaler()))

    if pipeline_steps:
        pipeline = make_pipeline(*[step for _, step in pipeline_steps])
        X = pipeline.fit_transform(features_flat)
        print(f'  Standardization applied')
    else:
        # No transformation needed, create identity pipeline
        pipeline = make_pipeline()  # Empty pipeline
        X = features_flat

    print('  No dimensionality reduction applied')
    
    transformer = {
        'method': 'none',
        'pipeline': pipeline
    }
    return X, transformer

def _load_destrieux_atlas(atlas_csv):
    df = pd.read_csv(atlas_csv)
    required_cols = ['name', 'hemisphere', 'lobe']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns in atlas CSV {atlas_csv}: {missing_cols}')
    return df

def _build_lobe_groups(ch_names, atlas_df):
    """
    Create a list of indicies of channels in ch_names for each group, where
    group corresponds to a given combination of (hemisphere. lobe).

    Return a list of arrays of indices per group, and a list of group labels
    constructed as 'hemisphere-lobe'
    
    """
    missing = [name for name in ch_names if name not in set(atlas_df['name'])]

    if missing:
        raise ValueError(f'{len(missing)} channel names missing from atlas')

    # Create a rec.array object of all combinations (hemisphere, lobe)
    group_keys = atlas_df[['hemisphere', 'lobe']].drop_duplicates().to_records(index=False)

    # Create mapping (hemisphere,lobe) -> pair #
    group_key_to_idx = {tuple(key): idx for idx, key in enumerate(group_keys)}

    # Initialize a list of idx of channels belonging to each group
    group_indices = [[] for _ in range(len(group_keys))]
    group_labels = [f'{key[0]}-{key[1]}' for key in group_keys] # Just string repr of keys

    # Now shorten group_labels, using only 1st letters of lobe names
    lobe_abbrev = {
        'FRONTAL': 'F',
        'PARIETAL': 'P',
        'OCCIPITAL': 'O',
        'TEMPORAL': 'T',
        'LIMBIC': 'L',
        'INSULAR': 'I',
    }

    group_labels = [
        # .get(part.upper(), part) just returns part if there is no mapping for that
        # part in lobe_abbrev dictionary
        '-'.join(
            '/'.join(lobe_abbrev.get(token.upper(), token) for token in part.split('/'))
            for part in label.split('-')
        )
        for label in group_labels
    ]

    # Get indices of channels in ch_names that belong to each group
    for chan_idx, name in enumerate(ch_names):
        # What is done: 1) find index value for specified ch name in atlas .csv;
        # 2) find the 0-based row number of this index value in the index
        # column. This is the channel number as listed in the atlas .csv
        # 3) get the (hemisphire, lobe) combination for this channel
        # 4) find group number for this key
        # 5) add this ch idx in ch_names list to this group
        row = atlas_df.index.get_loc(atlas_df.index[atlas_df['name'] == name][0])
        key = (atlas_df['hemisphere'].iloc[row],atlas_df['lobe'].iloc[row])
        group_idx = group_key_to_idx[key]
        group_indices[group_idx].append(chan_idx)

    # Convert a list of lists to a list of integer np.arrays
    group_indices = [np.array(idxs, dtype=int) for idxs in group_indices]

    # Check if there are empty groups (not really necessary to redo it every time, but still)
    empty_groups = [group_labels[i] for i, idxs in enumerate(group_indices) if len(idxs) == 0]
    if empty_groups:
        preview = ', '.join(empty_groups[:10])
        suffix = '...' if len(empty_groups) > 10 else ''
        raise ValueError(f'No channels found for lobe group(s): {preview}{suffix}')

    # Return list of arrays of ch indicies for each group, and corresponding
    # group labels
    return group_indices, group_labels

def _limit_to_lobes_spike_band(raw_features, config_dict):
    """
    Optionally restrict `raw_features` frequency axis for `to_lobes` reducer.

    Controlled by JSON keys:
      - to_lobes.limit_band (bool)
      - to_lobes.spike_band ([low_hz, high_hz])

    Returns:
        limited_features(ndarray): raw features with possibly reduced frequency axis
        freq_idx(ndarray|None): selected frequency indices when limiting is enabled
        reduced_freqs(list): frequency labels that correspond to returned features
    """
    to_lobes_config = config_dict.get('to_lobes', {}) or {}
    limit_band = bool(to_lobes_config.get('limit_band', False))

    freqs = np.asarray(config_dict.get('freqs', []), dtype=float)
    if freqs.size == 0:
        raise ValueError('to_lobes reducer requires frequency labels in config_dict["freqs"]')

    if freqs.shape[0] != raw_features.shape[2]:
        raise ValueError(
            'Mismatch between number of frequency labels and raw feature frequency dimension '
            f'({freqs.shape[0]} vs {raw_features.shape[2]})'
        )

    if not limit_band:
        return raw_features, None, freqs.tolist()

    spike_band = to_lobes_config.get('spike_band', None)
    if not isinstance(spike_band, (list, tuple)) or len(spike_band) != 2:
        raise ValueError(
            'When to_lobes.limit_band=true, to_lobes.spike_band must be a 2-item list/tuple [low_hz, high_hz]'
        )

    low_hz = float(spike_band[0])
    high_hz = float(spike_band[1])
    if high_hz < low_hz:
        low_hz, high_hz = high_hz, low_hz

    freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(freq_mask):
        raise ValueError(
            f'to_lobes.spike_band [{low_hz}, {high_hz}] selects no frequencies from available set'
        )

    freq_idx = np.flatnonzero(freq_mask)
    reduced_freqs = freqs[freq_mask].tolist()
    limited_features = raw_features[:, :, freq_mask, :]

    print(
        f'  to_lobes frequency limit enabled: kept {len(freq_idx)}/{len(freqs)} bins '
        f'in [{low_hz}, {high_hz}] Hz'
    )

    return limited_features, freq_idx, reduced_freqs

def _reduce_to_lobes(raw_features, group_indices):
    """
    Reduce raw_features dimensions by replacing ROIs with (hemisphere,lobe) pairs
    and assigning to such pairs averages of params over corresponding (hemisphere, lobe).
    Return reduced features array with shape `(nscans, ngroups, nfreq, nparms)`
    """
    nscans, _, nfreqs, nparms = raw_features.shape
    ngroups = len(group_indices)

    # Initialize reduced features
    reduced = np.zeros((nscans, ngroups, nfreqs, nparms), dtype=raw_features.dtype)

    for group_idx, chan_indices in enumerate(group_indices):
        # chan_indicies is a vector of idx of ROIs for each hemisphere/lobe group
        reduced[:, group_idx, :, :] = raw_features[:, chan_indices, :, :].mean(axis=1)

    return reduced

def _build_and_fit_to_lobes(raw_features, config_dict, standardize=False):
    """
    Build and fit 'to_lobes' dimensionality reduction method. In accordance with
    general interface requirements, it returns `nscans x nfeatures` matrix of 
    dim-reduced data, and `transformer` dictionary. In particular, the latter
    carries a ready to use `pipeline` object that can be applied to new data.

    Args:
        features_flat(ndarray): `shape (nscans, nfeatures)` - flattened features
        config_dict(dict): configuration dictionary (unused for this method)
        standardize(bool): whether to standardize features

    Returns:
        X(ndarray): `shape (nscans, nfeatures)` - flattened (and optionally standardized) features
        transformer(dict): fitted transformer dictionary with 'method' and 'pipeline' keys

    """
    ch_names = config_dict.get('ch_names')
    if not ch_names:
        raise ValueError('to_lobes reducer requires channel names in config_dict["ch_names"]')

    to_lobes_config = config_dict.get('to_lobes', {}) or {}
    atlas_csv = to_lobes_config.get('atlas_csv')

    if atlas_csv is None:
        atlas_csv = Path(__file__).with_name('destrieux_atlas_ordered.csv')
    else:
        atlas_csv = Path(atlas_csv)
        if not atlas_csv.is_absolute():
            atlas_csv = Path(__file__).parent / atlas_csv

    if not atlas_csv.exists():
        raise FileNotFoundError(f'Atlas CSV not found: {atlas_csv}')

    atlas_df = _load_destrieux_atlas(atlas_csv)
    group_indices, group_labels = _build_lobe_groups(ch_names, atlas_df)

    raw_features_limited, freq_idx, reduced_freqs = _limit_to_lobes_spike_band(raw_features, config_dict)
    reduced = _reduce_to_lobes(raw_features_limited, group_indices)
    nscans = reduced.shape[0]
    features_flat = reduced.reshape(nscans, -1)

    if standardize:
        pipeline = make_pipeline(StandardScaler())
        X = pipeline.fit_transform(features_flat)
        print('  Standardization applied')
    else:
        pipeline = None
        X = features_flat

    print(f'  Reduced to {len(group_indices)} lobe groups')

    transformer = {
        'method': 'to_lobes',
        'pipeline': pipeline,
        'group_indices': group_indices,
        'group_labels': group_labels,
        'atlas_csv': str(atlas_csv),
        'freq_idx': freq_idx,
        'freqs': reduced_freqs,
    }
    return X, transformer

def _apply_to_lobes(raw_features, transformer):
    freq_idx = transformer.get('freq_idx')
    if freq_idx is not None:
        raw_features = raw_features[:, :, freq_idx, :]

    group_indices = transformer['group_indices']
    reduced = _reduce_to_lobes(raw_features, group_indices)
    nscans = reduced.shape[0]
    features_flat = reduced.reshape(nscans, -1)

    pipeline = transformer.get('pipeline')
    if pipeline is None:
        return features_flat

    return pipeline.transform(features_flat)

def load_data(ss):
    """
    Load raw features (3D, not flattened) for specified scan IDs.

    Args:
        ss(obj): reference to this app object

    Returns:
        data(ndarray): `shape (nscans, nchans, nfreqs, nparms)` - the raw feature array.
            Contains distribution parameters for each channel, frequency, and scan.
            `nparms = 9` ('mean', 'median', 'std', 'skew', 'kurtosis',
            'ew_a', 'ew_c', 'ew_loc', 'ew_scale').
        chnames(list of str): channel names
        freqs(list): frequency labels
        parm_names(list of str): distribution parameters names

    """
    lst_IDs = ss.args['scan_ids']
    nscans = len(lst_IDs)
    data = None
    expand_sid = lambda sid: (ss.args['hospital'][sid[0]], sid[1])

    for i, sid in enumerate(lst_IDs):
        # Read the data, convert to numpy array, and drop last two columns
        tfd_fname = ss.tfd_pname(*expand_sid(sid))
        tfd = TFData.read(tfd_fname)                # Full TFData object
        fits = tfd.data.to_numpy()[:, :, :-2]       # distribution fit result: nchans x nf x nparms

        # The original list of parms:
        # 'mean', 'median', 'std', 'skew', 'kurtosis', 'ew_a', 'ew_c', 'ew_loc', 'ew_scale',
        # 'ew_fit_stat', 'ew_fit_pval'
        # The last two are fit results and are not included into features (but may be used
        # to filter out bad fits), making the nparm = 9

        if i == 0:
            ch_names = tfd.ch_names
            freqs = tfd.freqs
            parm_names = tfd.parm_names[:-2]
            nchans, nfreqs, nparms = fits.shape

            if ss.args['use_moments_only']:
                parm_names = parm_names[:5] # All up to kurtosis
                nparms = 5

            data = np.zeros((nscans, nchans, nfreqs, nparms))
            # end of initializations

        data[i, :, :, :] = fits[:,:,:nparms]

    return data, ch_names, freqs, parm_names

def list_src_ids(ss):
    """
    Return a list of tuples: (hospital #, Scan ID) corresponding to the _tfd.hdf5 files
    in the data folders for requested hospitals.
    """
    lst_ids = []
    for ih, hospital in enumerate(ss.args['hospital']):
        folder = ss.data_dir(hospital)      # Data folder corresponding to the selected hospital
                                            # and selected 'what'
        suffix = '_tfd' if ss.args['what'] == 'sensors' else  '_src_tfd'
        chop_off = -len(suffix)

        lst_ids += [(ih,p.stem[:chop_off]) for p in folder.glob('*_tfd.hdf5')]

    return lst_ids

def load_labels(ss, return_df = False):
    """
    Load labels from the database.

    Args:
        ss(obj): reference to this app object
        return_df(bool): flag to return full dataframe mapping the scan ID
            to all its classifications labels

    Returns:
        lbs(ndarray): vector of `nscans` labels, for requested classification
            (diagnosis), specified in JSON
        df_scan2labels(DataFrame): `shape (nscans, ncls+1), ncls=5` - dataframe with
            the first column with sIDs and `ncls` classification columns for each scan:  
            'Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi', 'Abnormality'
            (the names are actually set in the configuration JSON). `df_scan2labels``
            is **only returned when `return_df` flag is set**.

    NOTE: `ss.args['scan_ids']` is updated so that scans with no labels are no
    longer present

    """
    # Get scanID _> report ID mapping from eeg info database
    conn = sqlite3.connect(ss.eeg_info_db_pname)
    
    query = "SELECT ScanID, Hashed_ReportURN FROM 'EEG Metadata';"
    df = pd.read_sql_query(query, conn)

    # Keep only the scan IDs requested in `ss.args['scan_ids']`.
    sids = {id for _,id in ss.args['scan_ids']}     # Create a set of ids for fast search

    if sids:
        mask = df['ScanID'].isin(sids)

        if not mask.all():  # If something needs to be dropped from df
            df = df.loc[mask].reset_index(drop=True).copy()
            # Clean up temporaries and force garbage collection
            del mask
            del sids

    conn.close()

    # Now df has only sID -> repID mapping for requested sIDs

    # Get report ID -> labels mapping from the classifications DB
    conn = sqlite3.connect(ss.classifications_db_pname)

    '''
    # This creates a mess, because columns with spaces in name
    # should be put in quotes for SELECT to read right, but then
    # these columns are not found in DB:

    # Create a string with list of columns. Note that col names
    # without spaces should be used as is, while those with spaces
    # put in single quotes
    wrap_it = lambda s: '\''+s+'\'' if ' ' in s else s
    cols = ''
    cols = [wrap_it(s) for s in ss.args['classifications_cols']]
    cols_string = ','.join(cols)
    query = f"SELECT {cols_string} FROM 'classifications';"
    '''

    # So we'll simply use '*' for query
    query = f"SELECT * FROM 'classifications';"

    df_labels = pd.read_sql_query(query, conn)
    conn.close()
    # Now df_labels has mapping repID -> labels

    # Add physicians names to labels.
    reports_db_name = ss.args.get('reports_db')
    reports_db_pname = ss.db_root / reports_db_name

    if not reports_db_pname.exists():
        raise FileNotFoundError(f'reports_db not found: {reports_db_pname}')

    with sqlite3.connect(reports_db_pname) as reports_conn:
        reports_table = 'reports'
        df_physician = pd.read_sql_query(
            f"SELECT \"Hashed ID\", Physician FROM '{reports_table}';",
            reports_conn,
        )

    # Normalize and deduplicate before merge to avoid row multiplication.
    df_physician = (
        df_physician.rename(columns={'Hashed ID': 'Hashed_ReportURN'})
        .drop_duplicates(subset='Hashed_ReportURN', keep='first')
    )

    df_labels = df_labels.merge(df_physician, on='Hashed_ReportURN', how='inner')

    # We count on report ID column being the 1st in df_labels
    rep_id_col = df_labels.columns[0]

    # Force conversion to int of all cols except the report ID and Physician:
    convert_cols = ['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi', 'Abnormality']
    df_labels.loc[:, convert_cols] = df_labels.loc[:, convert_cols].astype(int)

    # Identify scan IDs from `df` that don't have corresponding
    # repID in `df_labels` and will be dropped by the inner join. 
    missing_scanids = set(df.loc[~df[rep_id_col].isin(df_labels[rep_id_col]), 'ScanID'].unique().tolist())

    if missing_scanids:
        print(f'{len(missing_scanids)} scan IDs have no corresponding reports and are not labeled')

    # Inner join df (scan->report mapping) with df_labels (report->classifications)
    # on rep_id_col, drop rows with any NaNs from the result, and
    # create a clean copy to ensure memory for dropped rows is released.
    try:
        df = df.merge(df_labels, on=rep_id_col, how='inner')
    except KeyError as e:
        raise KeyError(f"Column {rep_id_col} must be present in both DataFrames to perform join.") from e

    # Discard the report IDs
    df = df.drop(columns=rep_id_col)

    # Drop rows with any NaNs in the merged frame and make a fresh copy
    if df.isnull().values.any():
        df = df.dropna()

    df = df.reset_index(drop=True).copy()

    ids_to_keep = set(df['ScanID'].unique().tolist())   # Get an updated list of ScanIDs to work with (those that have labels)
    ss.args['scan_ids'] = [tpl for tpl in ss.args['scan_ids'] if tpl[1] in ids_to_keep] 

    # Scan IDs in df are not necessarily in the same order as scan IDs read
    # from the file system or set manually. Reorder df to that exact order.
    lst = [tpl[1] for tpl in ss.args['scan_ids']]

    if df['ScanID'].duplicated().any():
        raise ValueError("Duplicate ScanID entries found while building ordered labels.")

    # Reindex by ScanID to enforce requested order in the dataframe itself
    ordered_df = df.set_index('ScanID').reindex(lst)

    if ordered_df.isnull().values.any():
        # We are here when there is no index value for some elements in `lst`
        missing_ids = ordered_df.index[ordered_df[ss.args['target_label']].isnull()].tolist()
        raise ValueError(f"Missing labels for ScanID(s): {missing_ids}")

    # Keep ScanID as regular column and preserve order from `lst`
    df = ordered_df.reset_index()

    # Labels are now directly readable from the correctly ordered dataframe
    lbs = df[ss.args['target_label']].to_numpy()

    lbs -= 1  # Make labels 0-based

    if ss.args['ignore_confidence']:    # If ignoring confidence in labels
        lbs[lbs<=1] = 0         # 0,1 -> 0
        lbs[lbs>=2] = 1         # 1,2 -> 1
    else:
        # Drop elements with labels 1 and 2 (middle confidence values)
        keep_mask = (lbs != 1) & (lbs != 2)
        lbs = lbs[keep_mask]        # Drop from returned labels
        df = df.loc[keep_mask].reset_index(drop=True)  # Keep dataframe aligned with lbs

        # Drop corresponding scanIds from ss.args['scan_ids'] and from the form
        ss.args['scan_ids'] = [scan_id for i, scan_id in enumerate(ss.args['scan_ids']) if keep_mask[i]]
        print(f'Dropped {(~keep_mask).sum()} samples with uncertain labels (1, 2). Remaining: {len(lbs)}')

        # Replace label 3 with label 1
        lbs[lbs == 3] = 1

    # Optional physician-based filtering.
    # If physician list is empty or None, leave data unchanged.
    physician_filter = ss.args.get('physician')

    if physician_filter:
        # Convert everything to lower case, remove outer whitespace from names
        physician_set = {
            str(name).strip().casefold()
            for name in physician_filter
            if name is not None and str(name).strip()
        }

        # Get mask of records matching requested physicians, making same conversions
        keep_mask = (
            df['Physician']
            .fillna('')
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin(physician_set)
            .to_numpy()
        )
        dropped = int((~keep_mask).sum())

        if dropped:
            print(
                f"Dropped {dropped} samples not matching physician filter {sorted(physician_set)}. "
                f"Remaining: {int(keep_mask.sum())}"
            )

        df = df.loc[keep_mask].reset_index(drop=True)
        lbs = lbs[keep_mask]
        ss.args['scan_ids'] = [scan_id for i, scan_id in enumerate(ss.args['scan_ids']) if keep_mask[i]]

    if return_df:
        return lbs, df

    return lbs

def _runtime_xgboost_version():
    return str(getattr(xgboost, '__version__', 'unknown'))

def serialize_xgb_model_to_ubj_buffer(model):
    booster = model.get_booster() if hasattr(model, 'get_booster') else model
    try:
        model_bytes = booster.save_raw(raw_format='ubj')
    except TypeError as exc:
        raise RuntimeError(
            'Current XGBoost version does not support save_raw(raw_format="ubj") for in-memory UBJ export.'
        ) from exc

    model_bytes = bytes(model_bytes)
    if not model_bytes:
        raise RuntimeError('Failed to serialize XGBoost model to UBJ memory buffer')
    return model_bytes

def load_xgb_model_from_ubj_buffer(model_buffer):
    if model_buffer is None:
        raise KeyError('Missing XGBoost model buffer (full_model_data["model_ubj"])')

    model_bytes = bytes(model_buffer)
    if len(model_bytes) == 0:
        raise ValueError('XGBoost model buffer is empty')

    model = XGBClassifier()
    model.load_model(bytearray(model_bytes))
    return model

def verify_pickle_xgboost_version(payload, pkl_pname):
    if not isinstance(payload, dict):
        raise TypeError(f'Invalid classification pickle payload in {pkl_pname}: expected dict')

    saved_version = payload.get('xgboost_version')
    runtime_version = _runtime_xgboost_version()

    if saved_version is None:
        warnings.warn(
            f'Missing "xgboost_version" in classification pickle: {pkl_pname}. '
            'You may want to recompute and resave results with current pipeline.',
            RuntimeWarning,
        )
        return False

    if str(saved_version) != runtime_version:
        warnings.warn(
            f'XGBoost version mismatch for {pkl_pname}: '
            f'pickle has {saved_version}, runtime has {runtime_version}.'
            'You may want to recompute and resave results with current pipeline.',
            RuntimeWarning,
        )
        return False

    return True

