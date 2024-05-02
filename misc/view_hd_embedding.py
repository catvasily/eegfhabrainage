"""
*view_hd_embedding*: a utility to visualize high-dimenstional data by
embedding it into a 2D or 3D space by applying MDS, tSNE, UMAP
or similar algortithm, and plotting the embedded data.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import umap

_tSNE_args = {                      # Default parameter settings for tSNE embedding
    'perplexity': 30.0,
    'early_exaggeration': 12.0,
    'learning_rate': 'auto',
    'n_iter': 1000,
    'n_iter_without_progress': 300,
    'min_grad_norm': 1e-07,
    'metric': 'euclidean',
    'metric_params': None,
    'init': 'pca', 'verbose': 1,
    'method': 'barnes_hut',
    'angle': 0.5,
    'n_jobs': None
    }

def view_hd_embedding(data, *, taxonomy = None, method = None, nlow = 2, seed = None, embed_args = None,
        title = None, labels = None, show_plot = True, save_file = None, plot_args = None, plot_kwargs = None):
    """
    Embed a set of data points in high dimensional space in 2D or 3D space
    and plot the results.

    Args:
        data (ndarray of floats): shape (npoints, n_high_dims) the high dimensional data.
        taxonomy(ndarray of int or None): shape (npoints,) a mapping from point # to its category,
            which is an integer number. Taxonomy is only used for visualization (plotting),
            where different categories will be plotted with different colors.
        method (str or None): embedding method; one of 'MDS', 'tSNE', 'UMAP'. If not specified (None),
            'tSNE' will be applied by default.
        nlow (int): number of low dimensions: 2 or 3
        seed (int or None): if not None - a seed value for random generator. If not specified
            the embedding results may vary from one call to another
        embed_args(dict or None): a dictionary of parameters specific to the embedding method, with exclusion
            of the 'ncomponents' parameter which is already passed as 'nlow':
            'MDS' - see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html;  
            'tSNE' - see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html 
            'UMAP' - see https://umap-learn.readthedocs.io/en/latest/parameters.html
            For 'tSNE' method default parameter set will be provided if omitted (None).
        title (str or None): plot title
        labels (dict or None): labels for the categories encountered in taxonomy, in the form
            {..., k: 'label-for-category-k',...}
        show_plot (bool): flag to display the plotted figure
        save_file (str or None): if specified - pathname for the file to save the plot
        plot_args (dict or None): a dictionary with a minimal set of plotting parameters; if None -
            a default set will be used. This dictionary contains the following keys:
            'figsize' = (width, height) - figure size in inches  
            'colors' = list of standard color names (strings) to be used for plotting different categories  
            'dpi' = resolution of the saved figure
        plot_kwargs (dict or None): additional parameters to pass to matplotlib `scatter()` function

    Returns:
        A tuple (low_dim_data, fig), where
        low_dim_data (ndarray): shape (npoints, n_low_dims) the embedded
            low dimensional data
        fig (matploglib Figure): the figure object
    """
    if method is None:
        method = 'tSNE'

    if (embed_args is None) and (method == 'tSNE'):
        embed_args = _tSNE_args

    npoints = data.shape[0]

    if (taxonomy is not None) and (taxonomy.shape[0] != npoints):
        raise ValueError('Length of taxonomy array should match the number of data points')

    # Embed
    low_dim_data = do_embedding(data, method, nlow, seed, embed_args)  # n_points X nlow

    # Plot in low dimensions
    fig = plot_scatter(low_dim_data, taxonomy, title, labels, show_plot,
        save_file, plot_args, plot_kwargs)

    return low_dim_data, fig

def do_embedding(data, method, n_components, seed, embed_args):
    """
    Function to perform the actual embedding into 2D or 3D space.

    Args:
        data (ndarray of floats): shape (npoints, n_high_dims) the high dimensional data.
        method (str): embedding method; one of 'MDS', 'tSNE'
        n_components (int): number of low dimensions: 2 or 3
        seed (int or None): if not None - a seed value for random generator. If not specified
            the embedding results may vary from one call to another
        embed_args(dict): a dictionary of parameters specific to the embedding method, in
            addition to `n_components` argument

    Returns:
        low_dim_data (ndarray): shape (npoints, n_low_dims) the embedded
            low dimensional data
    """
    method = method.upper()

    if method not in ('MDS', 'TSNE', 'UMAP'):
        raise ValueError(f'Unknown embedding method {method} specified')

    if n_components not in (2, 3):
        raise ValueError('The number of low dimensions should be either 2 or 3')

    if embed_args is None:
        embed_args = {}

    if method == 'MDS':
        mds = manifold.MDS(n_components, random_state = seed, **embed_args)
        low_dim_data = mds.fit_transform(data)      # n_points X n_components
    elif method == 'TSNE':
        tsne = manifold.TSNE(n_components, random_state = seed, **embed_args)
        low_dim_data = tsne.fit_transform(data)
    elif method == 'UMAP':
        umap_fit = umap.UMAP(n_components = n_components, random_state = seed, **embed_args)
        low_dim_data = umap_fit.fit_transform(data)

    return low_dim_data

_plot_args = {
        'figsize': (16, 12),
        'colors': ['red', 'blue', 'green', 'black', 'violet', 'cyan',
            'magenta', 'gray', 'orange', 'brown', 'yellow'],
        'dpi': 300
        }

def plot_scatter(low_dim_data, taxonomy = None, title = None, labels = None, show_plot = True,
        save_file = None, plot_args = None, plot_kwargs = None):
    """
    Display a scatter plot of the embedded data.

    Args:
        low_dim_data (ndarray): shape (npoints, n_low_dims) the embedded
            low dimensional data
        taxonomy(ndarray of ints or None): shape (npoints,) a mapping from point # to its category,
            which is an integer number. If not supplied, all points will be plotted with the same
            color.
        title (str or None): plot title
        labels (dict or None): labels for the categories encountered in taxonomy, in the form
            {..., k: 'label-for-category-k',...}
        show_plot (bool): flag to display the plotted figure
        save_file (str or None): if specified - pathname for the file to save the plot
        plot_args (dict or None): a dictionary with a minimal set of plotting parameters; if None -
            a default set will be used. This dictionary contains the following keys:
            'figsize' = (width, height) - figure size in inches  
            'colors' = list of standard color names (strings) to be used for plotting different categories  
            'dpi' = resolution of the saved figure
        plot_kwargs (dict or None): additional parameters to pass to matplotlib `scatter()` function

    Returns:
        fig (matplotlib Figure)
    """
    if plot_args == None:
        plot_args = _plot_args

    npoints, ndims = low_dim_data.shape

    fig = plt.figure(figsize=_plot_args['figsize'])
    proj = '3d' if ndims == 3 else None 
    ax = fig.add_subplot(111, projection=proj)

    ax.grid(False)

    if taxonomy is not None:
        # Set color mapping
        types = np.unique(taxonomy)     # A list of all possible categories of points
        colors = plot_args['colors']

        if len(colors) < len(types):
            print('WARNING: color palette size is smaller than number of data categories.')
            print('Some categories will be plotted with identical colors')
            nextra = len(types) - len(colors)
            colors = colors + nextra*[colors[-1]]

        # Set labels
        labs = {i:str(i) for i in types}  # Default labels

        if labels is not None:
            for i in labels:            # Replace default label with a provided one
                labs[i] = labels[i]     # if available

        # Plot each category in turn
        for i, tp in enumerate(types):
            points = low_dim_data[taxonomy == tp]     # Subset of points for a category

            if plot_kwargs is None:
                ax.scatter(*points.T, c = colors[i], label = labs[tp])
            else:
                ax.scatter(*points.T, c = colors[i], label = labs[tp], **plot_kwargs)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    if ndims == 3:
        ax.set_zlabel('X3')

    plt.legend()

    if title is None:
        title = f'{ndims}-dimensional embedding'

    plt.title(title)
    plt.tight_layout()

    # Save the plot
    if save_file is not None:
        plt.savefig(save_file, dpi=plot_args['dpi'])

    if show_plot:
        plt.show()

    return fig

if __name__ == '__main__': 
    # Unit test
    from sklearn.datasets import load_digits

    data, target = load_digits(return_X_y=True)
    #method = 'UMAP'
    #method = 'tSNE'
    method = 'MDS'

    view_hd_embedding(data, taxonomy = target, method = method, nlow = 2, seed = None, embed_args = None,
        title = f'Digits dataset, method = {method}', labels = None, show_plot = True, save_file = 'qq.png', plot_args = None, plot_kwargs = None)

