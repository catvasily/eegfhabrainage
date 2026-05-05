"""
**Nested cross-validation utilities.**
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def nested_cv(estimator, X, Y, outer_cv_splitter, inner_cv_splitter,
              param_grid, scoring='accuracy', n_jobs=1, verbose=0):
    """
    Run nested cross-validation with explicit train/validation/test separation.

    Per outer fold:
      1) Split into outer-train and outer-test.
      2) Run GridSearchCV on outer-train using inner CV splits
         (inner-train/inner-validation) to select hyperparameters.
      3) Refit best model on full outer-train and predict on outer-test.

    Args:
        estimator: the classifier (AKA learner, sklearn-compatible
            estimator)e.g., XGBClassifier
        X(ndarray): feature matrix, shape (n_samples, n_features)
        Y(ndarray): label vector, shape (n_samples,)
        outer_cv_splitter: outer CV splitter providing train/test indices
        inner_cv_splitter: inner CV splitter used by GridSearchCV
        param_grid(dict|list): GridSearchCV parameter grid. Format:
            ```
            <parm_name>: <list-of-vals-to-try>
            ...
            <parm_name>: <list-of-vals-to-try>
            ```
            Total number of fits for all inner and outer folds will be
            equal to `P(l_i)*n_inner_cv*n_outer_cv`, where `P(l_i)`is a
            product of lengths of all lists in `param_grid`

        scoring(str|callable): GridSearchCV scoring metric
        n_jobs(int): parallel workers for GridSearchCV; set to -1 to use all
            available CPUs, set to -2 to use all but one CPU (computer
            will be more responsive)
        verbose(int): GridSearchCV verbosity

    Returns:
        outer_scores(ndarray): per-fold outer test accuracy, shape (n_outer_folds,)
        y_pred(ndarray): out-of-fold class predictions, shape (n_samples,)
        y_score(ndarray): out-of-fold positive-class score when available,
            shape (n_samples,). If no probability/decision output is available,
            values are zeros.
        y_proba(ndarray|None): out-of-fold class probabilities when available,
            shape (n_samples, n_classes). None if estimator has no predict_proba.
        fold_summaries(list[dict]): per-fold tuning and performance metadata
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2:
        raise ValueError(f'X must be 2D (n_samples, n_features); got shape {X.shape}')
    if Y.ndim != 1:
        raise ValueError(f'Y must be 1D (n_samples,); got shape {Y.shape}')
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'X and Y sample mismatch: {X.shape[0]} != {Y.shape[0]}')

    n_samples = X.shape[0]
    y_pred_all = np.zeros_like(Y, dtype=float)
    y_score_all = np.zeros(n_samples, dtype=float)
    y_proba_all = None  # Not initializing yet because classifier might not return
                        # probabilities - only scores

    outer_scores = []
    fold_summaries = []

    for ifold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, Y)):
        print(f'  Nested CV outer fold {ifold + 1}/{outer_cv_splitter.get_n_splits()}')

        # Split into a train and test data for current fold
        X_train = X[train_idx]
        X_test = X[test_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        # (Re-)Initialize the grid searcher
        grid = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv_splitter,
            refit=True,             # This refit flag setting ensures that after the
                                    # best parm set is found, the model is retrained
                                    # on the full training data (i.e. train+validation)
                                    # (ready for trying on the outer test fold data)
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Run CV for each combination of parms on current
        # train data using inner splitter settings
        grid.fit(X_train, Y_train)

        # Save the best classifier trained on the full inner
        # fold
        best_estimator = grid.best_estimator_

        # This is the prediction for the best parms set
        # for current fold
        y_pred_fold = best_estimator.predict(X_test)

        # Get probabilities and scores for the best combination
        # on current test data (i.e. the outer fold test samples)
        y_proba_fold = None
        y_score_fold = None

        if hasattr(best_estimator, 'predict_proba'):
            try:
                y_proba_fold = best_estimator.predict_proba(X_test)
                if y_proba_fold.shape[1] > 1:
                    y_score_fold = y_proba_fold[:, 1]
                else:
                    y_score_fold = y_proba_fold.ravel()
            except Exception:
                y_proba_fold = None

        if y_score_fold is None and hasattr(best_estimator, 'decision_function'):
            try:
                decision = best_estimator.decision_function(X_test)
                y_score_fold = decision.ravel() if np.ndim(decision) > 1 else np.asarray(decision)
            except Exception:
                y_score_fold = None

        if y_score_fold is None:
            y_score_fold = np.zeros(len(test_idx), dtype=float)

        fold_acc = accuracy_score(Y_test, y_pred_fold)
        outer_scores.append(fold_acc)

        y_pred_all[test_idx] = y_pred_fold
        y_score_all[test_idx] = y_score_fold

        if y_proba_fold is not None:    # If probabilities are in fact available
            if y_proba_all is None:
                y_proba_all = np.zeros((n_samples, y_proba_fold.shape[1]), dtype=float)
            y_proba_all[test_idx, :] = y_proba_fold

        fold_summaries.append({
            'fold': ifold,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'best_params': dict(grid.best_params_),
            'best_inner_score': float(grid.best_score_),
            'outer_test_accuracy': float(fold_acc),
        })

        print(f'    Best inner-CV score: {grid.best_score_:.4f}; outer-test accuracy: {fold_acc:.4f}')

    return np.asarray(outer_scores), y_pred_all, y_score_all, y_proba_all, fold_summaries
