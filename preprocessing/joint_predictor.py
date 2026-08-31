"""
Bayesean calculation of binary class correct prediction probabilities based on
predictions of N independent experts.

Assume that each expert predicted a binary class value of 0 or 1. Specifically, expert #i
predicted value Yi with probability of this prediction being true Pi, i=0,...,N-1.

Given {Yi, Pi}, and the priors, probability distribution P(Y | Y0,...Y_N-1) for Y=0,1
is calculated assuming expert's independence. See corresponding Notes document for
details.

AAM, June 7, 2006
"""

import numpy as np
from collections import namedtuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

import cls_calibrate

PRCurve = namedtuple('PRCurve', ['precision', 'recall', 'thresholds'])

LOGREG_DEFAULT_PARAMS = {
    'penalty': 'l2',
    'dual': False,
    'tol': 1e-4,
    'C': 1.0,
    'fit_intercept': True,
    'intercept_scaling': 1,
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

class JointBayes:
    """This class provides sklearn-style classifier wrapper to Naive-Bayes approach
    for ensemble binary class label prediction. The real workhorse is the class method
    ``bayes_joint_predict()`` which can be instead called directly if desired.

    Args:
        threshold_selection(str | None): threshold selection strategy used during
            training; same semantics as ``bayes_joint_predict()``.
        priors(pair of float): class priors for classes ``0`` and ``1``.

    Attributes:
        training_res(dict): results produced during ``fit()``.
        predict_res(dict): results produced during ``predict_proba()``.
    """

    def __init__(self, threshold_selection=None, priors=(0.5, 0.5)):
        self.threshold_selection = threshold_selection
        self.priors = priors
        self.training_res = {}
        self.predict_res = {}

    @staticmethod
    def _unpack_X(X):
        """Validate and unpack input container ``[y, y_proba]``."""
        if not isinstance(X, (list, tuple)):
            raise ValueError('X should be a list/tuple: [y, y_proba]')

        if len(X) != 2:
            raise ValueError('X should contain exactly 2 elements: [y, y_proba]')

        y, y_proba = X
        return y, y_proba

    def fit(self, X, y):
        """Fit threshold on labeled data and cache training diagnostics.

        Args:
            X(list | tuple): ``[y, y_proba]`` where ``y`` has shape
                ``(n_rec, n_exp)`` and ``y_proba`` has shape ``(n_rec, n_exp)``.
            y(array-like): true labels vector with shape ``(n_rec,)``.

        Returns:
            JointBayes: this estimator instance.
        """
        if y is None:
            raise ValueError('fit() requires true labels in argument y.')

        pred_y, pred_proba = self._unpack_X(X)
        self.training_res = self.bayes_joint_predict(
            pred_y,
            pred_proba,
            priors=self.priors,
            y_true=y,
            threshold_selection=self.threshold_selection,
            threshold=None,
        )
        self.predict_res = {}
        return self

    def predict_proba(self, X):
        """Predict class probabilities and labels using fitted threshold.

        Args:
            X(list | tuple): ``[y, y_proba]`` with model-expert outputs.

        Returns:
            np.ndarray: shape ``(n_rec, 2)`` with probabilities for classes
                ``0`` and ``1``.
        """
        if not self.training_res:
            raise ValueError('Estimator is not fitted: call fit() before predict_proba().')

        # This test is needed to ensure that fit() was already called, because in fact
        # our predict_proba() automatically also does the job of predict() - that is,
        # predicts the hard binary label
        if 'threshold' not in self.training_res:
            raise ValueError('Fitted results do not contain threshold.')

        pred_y, pred_proba = self._unpack_X(X)
        self.predict_res = self.bayes_joint_predict(
            pred_y,
            pred_proba,
            priors=self.priors,
            y_true=None,
            threshold_selection=self.threshold_selection,
            threshold=self.training_res['threshold'],
        )
        return self.predict_res['joint_probs']

    def predict(self, X, use_cached_prediction=True):
        """Predict binary class labels.

        Args:
            X(list | tuple): ``[y, y_proba]`` with model-expert outputs.
            use_cached_prediction(bool): if ``True`` and cached prediction exists,
                return it without recomputation.

        Returns:
            np.ndarray: shape ``(n_rec,)`` binary class labels.
        """
        if use_cached_prediction and self.predict_res:
            return np.asarray(self.predict_res['y_joint'], dtype=int)

        self.predict_proba(X)
        return np.asarray(self.predict_res['y_joint'], dtype=int)

    @classmethod
    def bayes_joint_probs(cls, rec_labels, rec_proba, priors=(0.5, 0.5)):
        """
        Estimate joint probs for classes 0 and 1 for a single sample, given
        ``n`` independent expert predictions. Use Naive Bayes classifier using
        'Odds" formulation. This allows to avoid the prior double counting problem.

        The expression is

        ``log(Odds(L1)|x_1,...,x_n)) = sum[log(Odds(L1|x_i))]-(n-1)log(Odds(L1))``

        where ``Odds = p/(1-p)``, ``x_i`` - features used by expert ``i``, ``L1`` - the
        label for the positive class.

        Args:
            rec_labels(list-like of int): shape (n,) - a list of class predictions made by ``n``
                experts for a single record.
            rec_proba(list-like of floats): shape (n,) - posterior probabilities that corresponding
                ``rec_labels`` are correct, that is ``p_i(L0,1|x_i), i=1,..,n``
            priors(pair of floats): prior probabilities of classes 0, 1

        Returns:
            joint_probs(pair of floats): estimated joint (total) probabilites for classes 0 and 1
                based on predictions ``rec_labels, rec_proba``.

        """
        rec_labels = np.asarray(rec_labels, dtype=int)

        if not np.all((rec_labels == 0) | (rec_labels == 1)):
            raise ValueError("The rec_labels array must contain only 0s and 1s")

        rec_proba = np.asarray(rec_proba, dtype=float)

        if not np.all((0.0 <= rec_proba) & (rec_proba <= 1.0)):
            raise ValueError("All rec_proba values must be in the interval [0, 1]")

        if rec_labels.shape != rec_proba.shape:
            raise ValueError("Sizes of rec_labels and rec_proba should match")

        if rec_labels.ndim != 1:
            raise ValueError("Both rec_labels and rec_proba should be 1-dimensional arrays")

        priors = np.asarray(priors, dtype=float)

        if not np.all((0.0 <= priors) & (priors <= 1.0)):
            raise ValueError("All priors values must be in the interval [0, 1]")

        if priors.shape != (2,):
            raise ValueError(
                "priors should be a vector of 2 probabilities - for class 0 and 1, respectively"
            )

        lodds = lambda p: np.log(p / (1 - p))

        loddsL1 = 0
        n = len(rec_proba)
        for y_i, p_i in zip(rec_labels, rec_proba):
            p = 1 - p_i if y_i == 0 else p_i
            loddsL1 += lodds(p)

        loddsL1 -= (n - 1) * lodds(priors[1])

        odds1 = np.exp(loddsL1)
        p1 = odds1 / (1 + odds1)
        p0 = 1 - p1

        return np.asarray([p0, p1])


    @classmethod
    def bayes_joint_predict(
        cls,
        y,
        y_proba,
        priors=(0.5, 0.5),
        y_true=None,
        threshold_selection=None,
        threshold=None,
    ):
        """
        For a set of records with their binary predicted labels and probabilities for
        those to be true, calculate optimized probabilites for each record
        to have labels 0 and 1 - that is, form a "joint" opinion and estimate its
        correctness. If true labels are available - then calculate PR curve and find
        optimal threshold for binary classification.

        IMPORTANTLY, PR curve and treshold estimation (that is, whenever the true labels are given)
        should be done on a separate held out subset. For actual predictions on new data the
        true labels should NOT be given; instead the threshold found with the held out data should be
        used for the binary class assignment if desired.

        Args:
            y(array of int): shape (n_rec,n_exp) - an array of class predictions made by
                ``n_exp`` experts for a dataset with ``n_rec`` records.
            y_proba(array of floats): shape (n_rec,n_exp) - probabilities that corresponding
                ``y``'s to be correct
            priors(pair of floats): prior probabilities of classes 0, 1
            y_true(list-like of int): shape (n_rec,) - true labels for this dataset; used
                to construct PR curve and estimate the threshold using selected method.
            threshold_selection(str | None): if ``None`` - then threshold = 0.5 will be used
                for binary class prediction; if 'F1' or 'f1' - then F1-optimized threshold will
                be found; other values are not supported yet.
            threshold(float | None): if given, then joint binary labels are calculated for
                each record based on given threshold. In this case PR curve and threshold
                selection calculations are skipped.

        Returns:
            res(dict): a dictionary with at least one key: ``'joint_probs'``, whose value is
                an array with shape (n_rec,2), containing calculated joint probabilities for
                classes 0,1 for each record. If threshold is given, then binary labels will be
                returned under key ``'y_joint'``. If ``y_true`` was provided, then ``'prcurve'``
                key will contain a named tuple of ``precision``, recall, thresholds`` as returned
                by ``precision_recall_curve()`` function; ``'threshold'`` key will contain the
                calculated threshold, and ``'y_joint'`` will contain binary predictions for the
                original dataset of ``n_rec`` records based on found threshold.

        """
        y = np.asarray(y, dtype=int)

        if y.ndim != 2:
            raise ValueError('y should be a 2D array with shape (n_rec, n_exp)')

        if not np.all((y == 0) | (y == 1)):
            raise ValueError('The y array must contain only 0s and 1s')

        y_proba = np.asarray(y_proba, dtype=float)

        if y_proba.ndim != 2:
            raise ValueError('y_proba should be a 2D array with shape (n_rec, n_exp)')

        if y.shape != y_proba.shape:
            raise ValueError('Shapes of y and y_proba should match')

        if not np.all((0.0 <= y_proba) & (y_proba <= 1.0)):
            raise ValueError('All y_proba values must be in the interval [0, 1]')

        n_rec = y.shape[0]
        y_joint_probs = np.zeros((n_rec, 2), dtype=float)

        for i in range(n_rec):
            y_joint_probs[i, :] = cls.bayes_joint_probs(y[i, :], y_proba[i, :], priors=priors)

        res = {'joint_probs': y_joint_probs}

        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError('threshold should be a float in [0, 1]') from exc

            if threshold < 0.0 or threshold > 1.0:
                raise ValueError('threshold should be in [0, 1]')

            res['y_joint'] = (y_joint_probs[:, 1] > threshold).astype(int)
            return res

        if y_true is None:
            return res

        y_true = np.asarray(y_true, dtype=int)

        if y_true.ndim != 1:
            raise ValueError('y_true should be a 1D array with shape (n_rec,)')

        if y_true.shape[0] != n_rec:
            raise ValueError('Length of y_true should match number of records in y and y_proba')

        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError('The y_true array must contain only 0s and 1s')

        precision, recall, thresholds = precision_recall_curve(y_true, y_joint_probs[:, 1])
        res['prcurve'] = PRCurve(precision=precision, recall=recall, thresholds=thresholds)

        if threshold_selection is None:
            threshold = 0.5
        else:
            if not isinstance(threshold_selection, str):
                raise ValueError('threshold_selection should be None or a string')

            method = threshold_selection.strip().lower()

            if method == 'f1':
                if thresholds.size == 0:
                    threshold = 0.5
                else:
                    pr = precision[:-1]
                    rc = recall[:-1]
                    denom = pr + rc
                    f1_vals = np.where(denom > 0, 2.0 * pr * rc / denom, 0.0)
                    best_idx = int(np.nanargmax(f1_vals))
                    threshold = float(thresholds[best_idx])
            elif method == '':
                threshold = 0.5
            else:
                raise ValueError('Unsupported threshold_selection. Supported values: None, "F1", "f1".')

        res['threshold'] = float(threshold)
        res['y_joint'] = (y_joint_probs[:, 1] > threshold).astype(int)
        return res


class JointLR:
    """Sklearn-style wrapper for LogisticRegression-based joint predictor.

    Args:
        joint_LR(dict | None): parameters from
            predict.summarize.joint_prediction.joint_LR.
    """

    def __init__(self, joint_LR=None):
        cfg = {} if joint_LR is None else dict(joint_LR)
        params = dict(LOGREG_DEFAULT_PARAMS)
        params.update(cfg)

        self.joint_LR = cfg
        self.lr_params = params
        self.model = LogisticRegression(**self.lr_params)
        self.calibrator = None
        self.cached_predict_proba = None    # Cached result of a call to predict_proba()
        self.threshold = 0.5                # Saved threshold for hard label predictions
        self.pr_curve = None                # Saved PR curve
        self.f1 = None

    @staticmethod
    def _unpack_X(X):
        """Validate and unpack input container ``[y, y_proba]``."""
        if not isinstance(X, (list, tuple)):
            raise ValueError('X should be a list/tuple: [y, y_proba]')

        if len(X) != 2:
            raise ValueError('X should contain exactly 2 elements: [y, y_proba]')

        y, y_proba = X
        return y, y_proba

    @staticmethod
    def _predictions_to_logits(y_pred, y_proba):
        """
        Helper that converts (label, probability) pairs from vectors
        y_pred[nrec,nexperts], y_proba[nrec,nexperts] into class 1 probs.It simply
        uses that p(C=0) = 1 - p(C=1). Then it returns the log-odds ln(p/(1-p)).
        Returned is an array of logits shaped(nrec,nexperts)
        """
        p = y_proba.copy()

        # Smallest step away from the boundaries
        epsilon = np.finfo(float).eps

        # Convert to positive class probs
        flip = (y_pred == 0)
        p[flip] = 1 - y_proba[flip]   # Now all p are class 1 probs

        # Clip for the logs to be safe
        p = np.clip(p, epsilon, 1.0 - epsilon)

        # Return logits
        return np.log(p / (1 - p))
        
    def fit(self, X, y_true):
        """Fit logistic-regression ensemble model.

        Args:
            X(list | tuple): [y_pred, y_proba] - training data. Here ``y_pred(array of ints)``,
                ``y_proba(array of floats)`` - predicted labels and their probabilities, both
                shaped ``(n_rec,n_exp)``
            y_true(vector of int): shape ``(n_rec,)``corresponding true labels.

        Returns:
            self (JointLR object): a reference to itself
        """
        y_pred, y_proba = self._unpack_X(X)    # y_pred = (n_train,); y_proba =  (n_train, n_exp)
        logits = self._predictions_to_logits(y_pred, y_proba)
        self.model.fit(logits, y_true)      # Let LR object to do the work
        
        # Reset results based on prior training
        self.calibrator = None
        self.cached_predict_proba = None    # clear cached results after a new fit
        self.threshold = 0.5                # Saved threshold for hard label predictions
        self.pr_curve = None                # Saved PR curve
        self.f1 = None
        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Args:
            X(list | tuple): [y_pred, y_proba] - new data with the same format as in fit().
        """
        y_pred, y_proba = self._unpack_X(X)    # y_pred = (n_train,); y_proba =  (n_train, n_exp)
        logits = self._predictions_to_logits(y_pred, y_proba)
        self.cached_predict_proba = self.model.predict_proba(logits)

        if self.calibrator is not None:
            calibrated_p1 = np.asarray(
                cls_calibrate.apply_calibrator(
                    calibration_obj=self.calibrator,
                    y_score=self.cached_predict_proba[:, 1],
                    method='betacal',
                )
            ).ravel()
            self.cached_predict_proba = np.column_stack((1.0 - calibrated_p1, calibrated_p1))

        return self.cached_predict_proba

    def predict(self, X, use_cached_prediction=True):
        """Predict binary class labels.

        Args:
            X(list | tuple): [y, y_proba].
            use_cached_prediction(bool): same argument contract as JointBayes.
        """
        if not use_cached_prediction or self.cached_predict_proba is None:
            self.cached_predict_proba = self.predict_proba(X)

        # !!! IMPORANT !!! There is no checks done if threshold selection was
        # already performed by calling pr_curve_threshold(), or a default
        # threshold is being used.
        return (self.cached_predict_proba[:,1] >= self.threshold).astype(int)

    def pr_curve_threshold(self, X, y_true, method = None, calibrate_scores=True):
        """
        Construct PR curve (should use held out data for X) and define threshold
        based on selected criterion

        Args:
            X(list | tuple): [y_pred, y_proba] - training data. Here ``y_pred(array of ints)``,
                ``y_proba(array of floats)`` - predicted labels and their probabilities, both
                shaped ``(n_rec,n_exp)``
            y_true(vector of int): shape ``(n_rec,)``corresponding true labels.
            method(str | None): threshold selection method, like "f1". If None - then default threshold
                0.5 will be used.
            calibrate_scores(bool): if True, calibrate class-1 scores before constructing
                the PR curve and selecting threshold.

        Returns:
            tuple(PRCurve, threshold): a tuple, whose 1st element is named tuple ('precision', 'recall', 'thresholds'),
                and 2nd element the threshold itself
        """
        # Construct the raw class-1 scores first so calibration is always fit
        # on the uncalibrated predictor output.
        y_pred, y_proba = self._unpack_X(X)
        logits = self._predictions_to_logits(y_pred, y_proba)
        p1 = self.model.predict_proba(logits)[:, 1]

        if calibrate_scores:
            self.calibrator, p1 = cls_calibrate.calibrate_with_method(
                method_name='betacal',
                y_score=p1,
                y_true=y_true,
                calibrate_cfg={},
                strict=True,
            )
        else:
            self.calibrator = None
        self.cached_predict_proba = np.column_stack((1.0 - p1, p1))

        precision, recall, thresholds = precision_recall_curve(y_true, p1)
        self.pr_curve = PRCurve(precision=precision, recall=recall, thresholds=thresholds)

        if method is not None:
            if method.lower() != 'f1':
                raise ValueError(f'Unsupported threshold selection method {method}')

            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

            # Find the index of the highest F1-score (excluding the last padded value)
            best_idx = np.argmax(f1_scores[:-1])
            self.threshold = thresholds[best_idx]
            self.f1 = f1_scores[best_idx]

        return self.pr_curve, self.threshold

