import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def data_to_pca(data, n_components = None, missing = None):
    """
    Utility function that converts a `m x n` matrix of `m` signals
    with `n` time points to `m x n` matrix of principal components
    time courses. The original data may contain NaNs; in this case
    each NaN value will be replaced with the mean value of corresponding
    signal.

    Args:
        data(ndarray): shape (m,n) `m` signal time courses with `n` time
            points.
        n_components (int): number of principle components to keep. If
            None - all components will be kept.
        missing (str): strategy for treating missing values. One of
            'mean', 'median' or 'most_frequent'. If not specified, then
            'mean' will be used.

    Return:
        data_pca(ndarray): (n_components x n) Time courses of the retained
            principal components.
        variances(ndarray): shape (n_components,) Variances of the retained
            components.
        ratios(ndarray): shape (n_components,) Ratios of variance explained
            by each component to the total variance of the data.
        means(ndarray): shape (m,) mean values for channels
        transform (ndarray): (n_components x m) PCA transformation matrix. 
            Each row is a unit-normalized m-dimensional PC vector.

    """
    m, n = data.shape

    if m > n:
        raise ValueError("For m x n data, m should be less or equal to n")

    if (n_components is None) or (n_components > m):
        n_components = m

    strategies = ('mean', 'median', 'most_frequent')

    if missing is None:
        missing = 'mean'
    elif not (missing in strategies):
        raise ValueError(f'The "missing" argument should be one of {strategies}')

    data_t = data.T

    # Impute missing values
    imputer = SimpleImputer(strategy = missing)
    data_t = imputer.fit_transform(data_t)

    # Initialize the PCA model
    pca = PCA(n_components = n_components)

    # Fit the PCA model to the imputed data and transform it
    pca_t = pca.fit_transform(data_t)   # n x n_components array, MATLAB ordering
    pca_t = np.ascontiguousarray(pca_t) # Now normal C-ordered array

    # Transpose back to get the result in the original n x m format
    data_pca = pca_t.T

    # Note that pca.components_ uses fortran storage, so we transform it
    # to C-storage on exit
    return data_pca, pca.explained_variance_, pca.explained_variance_ratio_, \
            pca.mean_, np.ascontiguousarray(pca.components_)


