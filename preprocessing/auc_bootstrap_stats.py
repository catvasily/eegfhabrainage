"""
For a sample with known true binary class labels and corresponding
positive class scores, estimate mean, STD and CI for the PR curve
AUC and F1 metrics. Use **stratified bootstrap** to properly deal with
highly unbalanced data.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score

def stratified_bootstrap_metrics(y_true, y_pred, threshold=None, n_bootstraps=2000, ci_level=0.95, seed = 54321):
    """
    Args:
        y_true (listlike of int): ``shape (n_samples,)`` true class labels, 0s and 1s
        y_pred(listlike of floats): ``shape (n_samples,)`` predicted class scores
        threshold(float): if provided, then F1 metrics stats will be also returned based on this
            threshold
        n_bootstraps(int): number of bootstrap resamples, default 2000
        ci_level(0<float<1): confidence interval witdh, default 95%
        seed(int): seed for random generator, default 54321

    Returns:
       results(dict): dictionary with keys ``'pr_auc'`` and ``'fi_score'`` (the latter if threshold was
            provided). Each key's value is also a dictionary with keys ``'mean'``,``'std'``,``'ci_lower'``,
            ``'ci_upper'``

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Separate indices by class
    idx_0 = np.where(y_true == 0)[0]
    idx_1 = np.where(y_true == 1)[0]
    
    n_0 = len(idx_0)
    n_1 = len(idx_1)
    
    bootstrapped_pr_aucs = []
    bootstrapped_f1s = []
    rng = np.random.default_rng(seed=seed)
    
    for _ in range(n_bootstraps):
        # Sample with replacement within each class independently
        sample_idx_0 = rng.choice(idx_0, size=n_0, replace=True)
        sample_idx_1 = rng.choice(idx_1, size=n_1, replace=True)
        
        # Combine the stratified indices
        boot_indices = np.concatenate([sample_idx_0, sample_idx_1])
        
        y_true_boot = y_true[boot_indices]
        y_pred_boot = y_pred[boot_indices]
        
        # 1. Calculate PR AUC
        precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_boot)
        pr_auc = auc(recall, precision)
        bootstrapped_pr_aucs.append(pr_auc)
        
        # 2. Calculate F1-score if threshold is provided
        if threshold is not None:
            # Binarize predictions based on the given threshold
            y_pred_binary = (y_pred_boot >= threshold).astype(int)
            f1 = f1_score(y_true_boot, y_pred_binary, zero_division=0)
            bootstrapped_f1s.append(f1)
            
    # -------------------------------------
    # Helper to calculate stats dictionary
    def get_stats(boot_values):
        boot_values = np.array(boot_values)
        lower_p = (1 - ci_level) / 2 * 100
        upper_p = (1 + ci_level) / 2 * 100
        return {
            "mean": np.mean(boot_values),
            "std": np.std(boot_values),
            "ci_lower": np.percentile(boot_values, lower_p),
            "ci_upper": np.percentile(boot_values, upper_p)
        }
    # -------- end helper -----------------
        
    results = {"pr_auc": get_stats(bootstrapped_pr_aucs)}
    
    if threshold is not None:
        results["f1_score"] = get_stats(bootstrapped_f1s)
        
    return results

import numpy as np
import scipy.stats as stats

def cohens_d_from_stats(m1, sd1, n1, m2, sd2, n2, alpha=0.05):
    """
    Calculates Cohen's d and its confidence interval from summary statistics of
    two sample sets, rather than using the sets directly.

    The confidence intervals for Cohen's D is based on the following
    analytical expression for its Standar Error of the sample Mean (which
    is in fact STD of the sample divided by sqrt(n)):

    ``SE=sqrt((n1+n2)/(n1 n2)+(1/2)D^2/(n1+n2))``
    
    Args:
        m1 (float): mean of group1
        sd1 (float): STD of group1
        n1 (int): samples size of group1
        m2 (float): mean of group2
        sd2 (float): STD of group2
        n2 (int): samples size of group2
        alpha(float): significance level for the CI (default 0.05 for 95% CI)

    Returns:
        results (dict): a dictionary with kesys ``'cohens_d', 'ci_lower',
            'ci_upper', 'standard_error'``.
        
    """
    # 1. Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * (sd1 ** 2) + (n2 - 1) * (sd2 ** 2)) / (n1 + n2 - 2))
    
    # 2. Calculate Cohen's d
    d = (m1 - m2) / pooled_sd
    
    # 3. Calculate the standard error of Cohen's d
    se_d = np.sqrt(((n1 + n2) / (n1 * n2)) + (d ** 2 / (2 * (n1 + n2))))
    
    # 4. Calculate the critical Z-score for the confidence interval
    # (Using normal distribution approximation, standard for Cohen's d CI)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    # 5. Calculate bounds
    ci_lower = d - (z_critical * se_d)
    ci_upper = d + (z_critical * se_d)
    
    return {
        "cohens_d": d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": se_d
    }


def pairwise_error_mcc(y_true, y_pred):
    """
    Calculates the pairwise Matthews Correlation Coefficient (MCC) of error vectors
    between n_cls classifiers.
    
    Args:
        y_true (ndarray): Array of shape (n_rec,) with true binary labels.
        y_pred (ndarray): Array of shape (n_rec, n_cls) with predicted binary labels.
    
    Returns:
        mcc_matrix(ndarray: Array of shape (n_cls, n_cls) containing pairwise MCCs.

    """
    # 1. Calculate binary error vectors for each classifier
    # y_true[:, None] broadcasts y_true from (n_rec,) to (n_rec, 1)
    # The result 'errors' will have shape (n_rec, n_cls) where 1 is an error and 0 is correct
    errors = (y_pred != y_true[:, None]).astype(np.float64) # We convert to float rightaway because means will be subtracted
    
    # 2. Calculate MCC pairwise using matrix operations. Mind that for binary vectors
    # MCC is the same as Pearson
    n_rec, n_cls = errors.shape
    
    # Centering...
    means = np.mean(errors, axis=0)
    centered_errors = errors - means
    
    # Compute the UNNORMALIZED covariance matrix between the error vectors (no division
    # by n_rec -1). Shape will be (n_cls, n_cls)
    covariance = np.dot(centered_errors.T, centered_errors)
    
    # Compute the STDs (unnormalized too)
    variances = np.sum(centered_errors ** 2, axis=0)
    std_devs = np.sqrt(variances)
    
    # Outer product of standard deviations creates the denominator matrix of shape (n_cls, n_cls)
    denominator = np.outer(std_devs, std_devs)
    
    # Avoid division by zero if a classifier has 0% or 100% error rate (constant error vector)
    with np.errstate(divide='ignore', invalid='ignore'):
        mcc_matrix = covariance / denominator
        # Replace NaNs (from zero division) with 0.0 as per standard correlation conventions
        mcc_matrix = np.nan_to_num(mcc_matrix, nan=0.0)
        
    return mcc_matrix

# Unit test
if __name__ == '__main__':
    # Generate mock data
    rng = np.random.default_rng(42)
    y_true = np.concatenate([np.zeros(900), np.ones(100)])
    y_pred = y_true + rng.normal(loc=0, scale=0.4, size=1000)
    y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

    # Run the updated bootstrap with a threshold
    stats = stratified_bootstrap_metrics(y_true, y_pred, threshold=0.5, n_bootstraps=2000, seed = 42)

    # Print PR AUC Results
    print(f"--- PR AUC Bootstrap Statistics ---")
    print(f"Mean:     {stats['pr_auc']['mean']:.4f}")
    print(f"STD:      {stats['pr_auc']['std']:.4f}")
    print(f"95% CI:   [{stats['pr_auc']['ci_lower']:.4f}, {stats['pr_auc']['ci_upper']:.4f}]\n")

    # Print F1-Score Results
    print(f"--- F1-Score (Threshold = 0.5) Statistics ---")
    print(f"Mean:     {stats['f1_score']['mean']:.4f}")
    print(f"STD:      {stats['f1_score']['std']:.4f}")
    print(f"95% CI:   [{stats['f1_score']['ci_lower']:.4f}, {stats['f1_score']['ci_upper']:.4f}]")

    """
    As tried on Jun 22, 2026 with this test code, the output is
    -----------------------------------------------------------

    --- PR AUC Bootstrap Statistics ---
    Mean:     0.8582
    STD:      0.0287
    95% CI:   [0.7990, 0.9105]

    --- F1-Score (Threshold = 0.5) Statistics ---
    Mean:     0.5628
    STD:      0.0200
    95% CI:   [0.5245, 0.6013]

    """

