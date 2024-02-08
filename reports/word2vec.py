"""
*word2vec*: utils to construct vector embeddings for the words in reports, 
for reports themselves ('doc2vec'), and also for clustering these vectors
and visualizing the results.
"""
import sys
import os.path as path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from scipy.spatial.distance import cdist
from sklearn import manifold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, BisectingKMeans, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from reports_db import ReportsDB
from create_db import add_db_tables

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

# These are only needed for benchmarking
sys.path.append(path.dirname(path.dirname(__file__))+ "/misc")
from utils import timeit

def word2vec(ss):
    """
    Train word2vec model on the reports data and construct the embedding.
    A subset of reports to work on is determined by keys `'select_physicians'`,
    `'select_hospitals' of the `INPUT_JSON_FILE`. Word2vec-specific input
    arguments are given under a key `'word2vec'`.

    The results are saved in the model file
    whose name is controlled by 'w2v_model_file_name key'. In
    fact, this key sets the file suffix (i.e. 'word2vec.model'), while the full file name is defined
    by the subset of reports used and will be like
    `<...>_word2vec.model`. For example, if all reports were used to generate
    the model, the file name will be `all_word2vec.model`. 

    """
    # The import statement is moved here to avoid 
    # a 'circular impoit'  error
    from process_reports import get_selected_reports

    # Load all/subset of reports
    if ss.db is None:
        ss.db = ReportsDB(ss.reports_db_pathname)

    reports = get_selected_reports(ss)  # This is a SET of reports

    # Preprocess reports
    parsed_reps = parse_reports(ss, reports)

    # Train the model
    model = gensim.models.Word2Vec(
        parsed_reps,
        workers = num_workers(), **ss.args['word2vec']['model_settings'])
    print('word2vec: done training, loss = {:.4e}'.format(model.get_latest_training_loss()))

    # Save it
    model_file = path.join(ss.args["reports_db_dir"],  ss.reports_for + \
            '_' + ss.args['word2vec']['w2v_model_file_name'])
    model.save(model_file)
    print(f'word2vec: model saved to file {model_file}')

def parse_reports(ss, reports):
    """
    Tokenize reports and remove stop words if requested in the input
    JSON file. Tokenizing is done using gensim's `simple_preprocessing()`
    utility.

    Args:
        ss: reference to this app object
        reports(iterable of str): oritinal reports as single strings

    Returns:
        parsed(list of lists of str): list of reports, each being a list of
            tokens
    """
    kwargs = ss.args['word2vec']['preproc_args']
    parsed_reps = []

    for rep in reports:
        parsed = gensim.utils.simple_preprocess(rep, **kwargs)

        if ss.args['ignore_stop_words']:
            parsed = [w for w in parsed if w not in ENGLISH_STOP_WORDS]

        parsed_reps.append(parsed)

    return parsed_reps

def doc2vec(ss):
    """
    Train gensim doc2vec model on the reports data and construct a vector
    embedding for selected reports. The results are saved in the model file
    whose name is controlled by 'd2v_model_file_name' key. In
    fact, this key sets the file suffix (i.e. 'doc2vec.model'), while the
    full file name is defined by the subset of reports used and will be like
    `<...>_doc2vec.model`. For example, if all reports were used to generate
    the model, the file name will be `all_doc2vec.model`.

    A subset of reports to work on is determined by keys `'select_physicians'`,
    `'select_hospitals' of the `INPUT_JSON_FILE`. Doc2vec-specific input
    arguments are given under a key `'doc2vec'`.
    """
    # The import statement is moved here to avoid 
    # a 'circular impoit'  error
    from process_reports import get_selected_tagged_reports

    # Load all/subset of reports
    if ss.db is None:
        ss.db = ReportsDB(ss.reports_db_pathname)

    # Get {(hid, rep)} set
    tagged_set = get_selected_tagged_reports(ss)

    # Tokenize the documents
    # Now tagged_set will become a list as sets cannot contain lists as elements
    tagged_set = [(hid, parse_reports(ss, {rep})[0]) for hid, rep in tagged_set]

    # Create a list of TaggedDocument objects
    lst_TDocs = [TaggedDocument(words=rep, tags=[hid]) for hid, rep in tagged_set]

    # Create and train the Doc2Vec model
    model = Doc2Vec(workers = num_workers(), seed = ss.seed, **ss.args['doc2vec']['model_settings'])
    model.build_vocab(lst_TDocs)
    print('doc2vec: done building vocabulary')
    model.train(lst_TDocs, total_examples=model.corpus_count, epochs=model.epochs)
    final_loss = model.get_latest_training_loss()   # Not implemented, actually; always 0
    print('doc2vec: done training, loss = {:.4e}'.format(final_loss))

    model_file = path.join(ss.args["reports_db_dir"],  ss.reports_for + \
            '_' + ss.args['doc2vec']['d2v_model_file_name'])
    model.save(model_file)
    print(f'doc2vec: model saved to file {model_file}')

def num_workers():
    """
    Return number of workers to use depending on number of available
    CPUs
    """
    return max(int(mp.cpu_count()/2)-2, 2)

def show_similar_words(ss):
    """
    Interactively show similar words, using a pretrained word2vec model.
    """
    model_file = input('Model file name: ')
    model_file = path.join(ss.args["reports_db_dir"], model_file)

    model = gensim.models.Word2Vec.load(model_file)

    while (True):
        word = input('Word: ')

        if not len(word):
            break

        try:
            df = pd.DataFrame.from_records(model.wv.most_similar(word, topn=5),
                    columns = ['Word', 'Score'])
            print(f'{word}:')
            print(df)
        except KeyError:
            print(f'Word "{word}" not present in the vocabulary')

def show_similar_reports(ss):
    """
    Interactively show similar words, using a pretrained doc2vec model.
    """
    model_file = input('Model file name: ')
    model_file = path.join(ss.args["reports_db_dir"], model_file)

    model = Doc2Vec.load(model_file)

    if ss.db is None:
        ss.db = ReportsDB(ss.reports_db_pathname)

    all_ids = set(ss.db.get_all_reps().keys())

    while (True):
        hid = input('Hash first few characters: ')

        if not len(hid):
            break

        # Get full hashed ID
        for h in all_ids:
            if h.find(hid) != -1:
                hid = h
                break

        try:
            df = pd.DataFrame.from_records(model.dv.most_similar(hid, topn=5),
                    columns = ['Hash ID', 'Score'])
            print(f'{hid}:')
            print(df)
        except KeyError:
            print(f'Report with hash "{hid}" not found')

def cluster_reports(ss):
    """
    Cluster report vectors passed as `ss.data` array with a shape
    `(nvectors, ndimensions)`.

    As a result, sets `ss.labs2idx` attribute, which is a mapping  
    `cluster_label -> list of cluster point numbers`  
    where cluster label is an integer starting with 0, and point numbers
    correspond to the row numbers in `ss.data` array.
    """
    get_doc_vectors_for_physicians(ss, 'cluster_reports')
    # Now ss.data is set in accordance with selected physicians
    # ss.lst_hid carries hashed IDs of the reports in `ss.data`
    # ss.phys2idx(dict): mapping physician -> list of indicies into `ss.lst_hid`
    # and `ss.data`.

    method = ss.args['cluster_reports']['method']

    if method == 'KMeans':
        ss.kmeans = KMeans(random_state = ss.seed,
                **ss.args['cluster_reports']['KMeans_settings'])
        labels = ss.kmeans.fit_predict(ss.data)  # (nsamples,) array of cluster indecies 
        ss.cluster_centers = ss.kmeans.cluster_centers_
        ss.n_clusters = ss.cluster_centers.shape[0] 
    elif method == 'BisectingKMeans':
        ss.bkmeans = BisectingKMeans(random_state = ss.seed,
                **ss.args['cluster_reports']['BisectingKMeans_settings'])
        labels = ss.bkmeans.fit_predict(ss.data)  # (nsamples,) array of cluster indecies 
        ss.cluster_centers = ss.bkmeans.cluster_centers_
        ss.n_clusters = ss.cluster_centers.shape[0] 
    elif method == 'HDBSCAN':
        ss.hdbscan = HDBSCAN(**ss.args['cluster_reports']['HDBSCAN_settings'])
        labels = ss.hdbscan.fit_predict(ss.data)  # (nsamples,) array of cluster indecies 
        ss.cluster_centers = ss.hdbscan.centroids_
        # NOTE: for HDBSCAN there also clusters labeled -1 (noise)
        # In general case, may be labels -2 (infinite elements) and -3 (missing data)
        # n_clusters only includes non-negative labels
        ss.n_clusters = ss.cluster_centers.shape[0] 
    else:
        raise ValueError('Unknown clustering method specified')

    # Create labels -> points mapping
    span = np.arange(len(labels))       #0,1,...,npoints-1

    # For each label, provide a list of corresponding point indices
    ss.labs2idx = {l:span[labels == l] for l in range(ss.n_clusters)}
    get_cluster_centers(ss)
    save_clustering_results(ss, labels)

    print(f'\n{method} clustering for {ss.n_clusters} clusters done.')
    print('Cluster center reports hashed IDs:')
    for l in ss.labs2idx:
        print(f'{l}: {ss.cluster_center_hid[l]}')
    print('\n')

    # Silhouette Score results seem to be misleading - so don't print those
    #print('Silhouette Score: {}'.format(silhouette_score(ss.data, labels)))

    print('Calinski-Harabasz Index: {}\n'.format(calinski_harabasz_score(ss.data, labels)))

def clustering_results_tbl_name(ss):
    """
    Return a name of the clustering results column in the clustering
    results database. This name is determined by the clustering method and
    the number of requested clusters (both specified in the JSON file), and may be
    something like 'KMeans results, 3 clusters'
    """
    return '"' + ss.args['cluster_reports']['method'] + \
            ' results, {} clusters'.format(ss.n_clusters) + '"' 

def save_clustering_results(ss, labels):
    """
    Add clustering results to the public `clustered_reports` database.
    """
    clustering_db_file = path.join(ss.args['reports_db_dir'],
            ss.args['clustered_reports_db_name'])

    # Table names
    results_tbl = clustering_results_tbl_name(ss)
    centers_tbl = '"Centroids for ' + ss.args['cluster_reports']['method'] + \
            ', {} clusters'.format(ss.n_clusters) + '"' 
    
    # Get table descriptions and rename the tables in accordance
    # with the method
    db_tables = ss.args['cluster_reports']['output_tables'] 
    db_tables[results_tbl] = db_tables['results']
    db_tables[centers_tbl] = db_tables['centers']
    del db_tables['results']
    del db_tables['centers']

    # Initialize tables in DB
    cursor = add_db_tables(clustering_db_file, db_tables,
            return_cursor = True)
    conn = cursor.connection

    # Write data to db
    for tbl in db_tables:
        if len(db_tables[tbl]) > 2:
            sql = f'INSERT INTO {tbl} VALUES (?, ?, ?, ?)'
            reps_dict = ss.db.get_reps_by_ids(ss.lst_hid)   # Dictionary hid:rep
            reps = [reps_dict[hid] for hid in ss.lst_hid]   # Get reps in the same order as lst_hid
            nrecords = len(labels)
            # NOTE: if one uses just 'l' instead of 'l.item()', the label is written to the DB
            # as a binary BLOB, not as a digit
            cursor.executemany(sql, zip(ss.lst_hid, [l.item() for l in labels], reps, nrecords*['']))
        else:
            # This is the cluster centers table
            sql = f'INSERT INTO {tbl} VALUES (?, ?)'
            cursor.executemany(sql, zip(range(ss.n_clusters), ss.cluster_center_hid))

    conn.commit()
    conn.close()

def get_cluster_centers(ss):
    """
    Return nominal cluster center locations in the feature space,
    and real data points closest to the cluster centers. Returns results
    as attributes of the `ss` object, as shown below:

    Returns:
        ss.cluster_centers (ndarray): `(nclusters, ndim)` nominal cluster center
            locations in the feature space
        ss.cluster_center_idx (ndarray of ints): `(nclusters, )` indicies of real data
            points closest to the corresponding nominal center locations.
        ss.cluster_center_hid (ndarray of np.str_): `(nclusters, )` hashed IDs of the
            reports closest to corresponding cluster centers.
    """
    nclusters = ss.cluster_centers.shape[0]
    ss.cluster_center_idx = np.zeros(nclusters, dtype=int)
    ss.cluster_center_hid = np.empty(nclusters, dtype=f'U{len(ss.lst_hid[0])}')

    for l in ss.labs2idx:
        cluster_points = ss.data[ss.labs2idx[l],:]
        center = ss.cluster_centers[l,:]
        distances = cdist(cluster_points, center[np.newaxis, :])    # npts x 1
        idx_in_cluster = np.argmin(distances, axis=0)[0]  # argmin returns 1 x 1 *array* of ints
        ss.cluster_center_idx[l] = ss.labs2idx[l][idx_in_cluster]   # map idx in cluster to global idx
        ss.cluster_center_hid[l] = ss.lst_hid[ss.cluster_center_idx[l]]

def set_cluster_codes(n_clusters):
    """
    Asks user to identify cluster type for each label, and returns a
    list mapping each label to a corresponding code. Currently, there
    are three specific codes defined: 'N' for normal cluster, 'A' for
    abnormal cluster and 'M' for mixed cluster. Codes for remaining
    clusters (if any) match their labels.

    Args:
        n_clusters (int): as is

    Returns:
        lst_codes (list of str): list of symbolic codes for labels
    """
    if n_clusters == 2:
        while(True):
            normal = int(input('Which label was assigned to NORMAL cluster (0 or 1)? '))

            if normal not in (0,1):
                print('Invalid label, only 0 or 1 are allowed')
            else:
                break

        abnormal = int(not normal)
    elif n_clusters >= 3:
        while(True):
            normal = int(input('Which label was assigned to NORMAL cluster? '))
            allowed = list(range(n_clusters))

            if normal not in allowed:
                print(f'Invalid label, only {allowed} are allowed')
            else:
                allowed.remove(normal)
                abnormal = int(input('Which label was assigned to ABNORMAL cluster? '))

                if abnormal not in allowed:
                    print('Invalid label, let\'s start over...')
                else:
                    allowed.remove(abnormal)

                    if n_clusters == 3:
                        mixed = allowed[0]
                        break
                    else:
                        mixed = int(input('Which label was assigned to MIXED cluster? '))

                        if mixed not in allowed:
                            print('Invalid label, let\'s start over...')
                        else:
                            allowed.remove(mixed)
                            break

    # At this point, normal, abnormal and mixed clusters are determined,
    # and uninterpreted labels are listed in 'allowed'

    # These printouts are for debug only
    #print(f'Assignment: normal {normal}, abnormal {abnormal}')

    #if n_clusters >=3:
    #    print(f'mixed {mixed}')

    #if n_clusters > 3:
    #    print(f'Unassigned: {allowed}')
    # End debut printout

    lst_codes = [str(i) for i in range(n_clusters)]
    lst_codes[normal] = 'N'
    lst_codes[abnormal] = 'A'

    if n_clusters >= 3:
        lst_codes[mixed] = 'M'

    return lst_codes

def clustering_info_to_reports_db(ss):
    """
    Update the processed reports database, specified by the
    'reports_db_name' key in the input JSON, with clustering
    results that where previously saved in a database pointed to by
    the 'clustered_reports_db_name' key. As cluster labels are assigned in
    no specific order, the user will be requested to manually enter
    which label corresponds to which symbolic cluster code.

    Specifically, if there are three clusters, the user will be requested to provide
    label numbers for normal, abnormal and mixed clusters. Those will be assigned
    codes 'N', 'A' and 'M', repsectively. If there are two clusters only,
    then just 'N' and 'A' cluster codes will be used. In all other cases the user will
    need to specify labels for 'N', 'A' and 'M' codes as before, while codes for all other labels
    will be automatically set equal to the label numbers themselves.
    """
    n_clusters = int(input('Target number of clusters used at the clustering step: '))

    if n_clusters <= 1:
        print('ERROR: The number should be larger than 1')
        sys.exit()

    ss.n_clusters = n_clusters

    # Define cluster codes for labels
    lst_codes = set_cluster_codes(n_clusters)

    """
    The update from clustering DB to results DB may be done
    via these statements:

    ATTACH DATABASE 'clustered_reports.db' AS db2;
    UPDATE reports
    SET 'Cluster code' = (
        SELECT Cluster
        FROM db2.'KMeans results, 3 clusters'
        WHERE db2.'KMeans results, 3 clusters'.'Hashed ID' = reports.'Hashed ID'
    )
    WHERE EXISTS (
        SELECT 1
        FROM db2.'KMeans results, 3 clusters'
        WHERE db2.'KMeans results, 3 clusters'.'Hashed ID' = reports.'Hashed ID'
    );

    For some reason, this is a very slow operation (takes ~ 2 min). A faster way is
    to read in the clustering results first, then update the results DB for each
    affected record.
    """

    # Get data from the clustering results DB 
    clustering_db_file = path.join(ss.args['reports_db_dir'],
            ss.args['clustered_reports_db_name'])

    try:
        cdb_conn = sqlite3.connect(clustering_db_file)
    except sqlite3.Error as e:
        print(e)
        sys.exit()

    cdb_cursor = cdb_conn.cursor()
    results_tbl = clustering_results_tbl_name(ss)

    # Get hashed ID, labels column names
    col_hid, col_label = ss.args['cluster_reports']['output_tables']['results'][:2]
    col_hid = col_hid[0]
    col_label = col_label[0]

    # The call is like: SELECT "Hashed ID", Cluster FROM 'KMeans results, 3 clusters';
    sql = f'SELECT {col_hid}, {col_label} FROM {results_tbl}'
    rows = np.array(list(cdb_cursor.execute(sql)))  # array of 64-char strings [[hid, label], ..., [hid, label]]
    
    cdb_conn.close()

    # Set cluster codes
    for i, code in enumerate(lst_codes):
        idx = (rows[:,1] == str(i))
        rows[idx,1] = code

    # Update the processed reports DB. The SQL call is like
    # UPDATE <tbl> SET column1 = value1, column2 = value2 WHERE condition
    tbl = ss.args['cluster_reports']['reports_db_info']['results_tbl']
    hid_col = ss.args['cluster_reports']['reports_db_info']['hid_col']
    cluster_col = ss.args['cluster_reports']['reports_db_info']['cluster_col']
    sql = f'UPDATE {tbl} SET {cluster_col} = ? WHERE {hid_col} = ?'

    close_db = False    # Flag to close the reports DB when done

    if ss.db is None:
        ss.db = ReportsDB(ss.reports_db_pathname)
        close_db = True

    ss.db.cursor.executemany(sql, rows[:,[1,0]])
    ss.db.conn.commit()

    if close_db:    # Close database if it was not originally open
        ss.db.conn.close()

def get_doc_vectors_for_physicians(ss, for_what):
    """
    Load doc2vec model file, select only data vectors
    for specified physicians. If total number of these
    vectors is larger than `max_reports` - subsample
    the data so that the total equals `max_reports`.

    This function returns results by setting corresponding
    attributes to the application object `ss` as shown below:

    Args:
        ss: reference to this app object
        for_what (str): either `'cluster_reports'` or `'plot_reports'`;
            the app step where doc2vec model is being used.

    Returns:
        ss.data(ndarray): `(npoints, ndim)` report vectors
        ss.lst_hid(array of str): hashed IDs of the reports in `ss.data`
        ss.phys2idx(dict): mapping physician -> list of indicies into `ss.lst_hid`
            and `ss.data` for each physician.
    """
    if for_what not in ('cluster_reports', 'plot_reports'):
        raise ValueError('Invalid "for_what" argument specified')

    model_file = path.join(ss.args["reports_db_dir"],
            ss.args[for_what]['d2v_model_file_name'])
    model = Doc2Vec.load(model_file)

    lst_hid = np.array(model.dv.index_to_key)
    data = model.dv.get_normed_vectors()

    # Construct a dictionary physician -> list of indecies
    # of elements in 'lst_hid' corresponding to each physician
    phys2idx = get_ids_for_physician(ss, lst_hid)
    n_hid = sum([len(v) for v in phys2idx.values()])

    # Get only subset of the data for each requested physician
    # if max allowed number of reports to process is smaller
    # than the total number of reports for those physicians
    if n_hid > ss.args['max_reports']:
        phys2idx = subsample_idx(ss, phys2idx, ss.args['max_reports'])

    # Now subsample everything in accordance with phys2idx
    lst_idx = np.concatenate(list(phys2idx.values()))
    lst_hid = lst_hid[lst_idx]
    data = data[lst_idx]

    # Update the mapping - now this becomes a straight consecutive indexing
    lst_idx = np.arange(len(lst_idx))
    lst_lens = [len(v) for v in phys2idx.values()]
    new_mapping = split_list(lst_idx, lst_lens)
    phys2idx = {k:new_mapping[i] for i, k in enumerate(phys2idx)}

    # Return:
    ss.data = data
    ss.lst_hid = lst_hid
    ss.phys2idx = phys2idx

def plot_reports(ss):
    """
    Perform low-dim embedding of the reports vectors produced by 
    `doc2vec` and display reports distribution in 2D or 3D space.
    """
    plot_title = lambda n: str(ss.args['select_physicians'])[1:-1] + ', ' +\
            ss.args['plot_reports']['low_dim_embedding'] + f' {n}D embedding'

    save_file = lambda n: str(ss.args['select_physicians'])[1:-1] + ', ' +\
            ss.args['plot_reports']['low_dim_embedding'] + f' {n}D embedding'

    if ss.data is None:
        get_doc_vectors_for_physicians(ss, 'plot_reports')

    if ss.args['plot_reports']['colors_for'] == 'physicians':
        clr2idx = ss.phys2idx
    elif ss.args['plot_reports']['colors_for'] == 'clusters':
        clr2idx = ss.labs2idx
    else:
        raise ValueError('Unrecognized value for "plot_reports"/"colors_for" specified')

    # Perform 2D or 3D embedding on sub-sampled data
    low_dim_data = low_dim_embedding(ss, ss.data)  # n_points X n_components

    # Scatter plot
    embedding = ss.args['plot_reports']['low_dim_embedding']
    ndims = ss.args["plot_reports"][embedding]["n_components"]
    fig = plt.figure(figsize=(16, 12))
    proj = '3d' if ndims == 3 else None 
    ax = fig.add_subplot(111, projection=proj)

    ax.grid(False)

    # Set color mapping. Use colors from 'phys_colors' table
    # whether we are actually plotting different physicians or
    # something else (i.e. clusters)
    phys2col = ss.args['plot_reports']['phys_colors']

    if len(clr2idx) <= len(phys2col):
        # Use colors from the start of the table if possible
        colors = list(phys2col.values())
    else:
        colors = list()
        for ph in clr2idx:
            colors.append(phys2col[ph] if ph in phys2col else phys2col['Other'])

    for i,ph in enumerate(clr2idx):
        points = low_dim_data[clr2idx[ph]]     # Subset of points for a physician
        ax.scatter(*points.T, c = colors[i], label = ph,
                **ss.args['plot_reports']['scatter_plot_args'])

    # Plot cluster centers with larger size markers
    if ss.args['plot_reports']['colors_for'] == 'clusters':
        points = low_dim_data[ss.cluster_center_idx]
        kwargs = ss.args['plot_reports']['scatter_plot_args'].copy()
        kwargs['s']*=16
        kwargs['alpha']=1
        ax.scatter(*points.T, c = colors[:len(points)], **kwargs)

    ax.set_title(f'Reports: {ndims}D {embedding} embedding')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    if ndims == 3:
        ax.set_zlabel('X3')

    plt.legend()
    plt.title(plot_title(ndims))
    plt.tight_layout()

    # Save the plot
    save_file = plot_title(ndims).replace(',', '')
    save_file = save_file.replace(' ','_')
    save_file = save_file.replace('\'','') + f'_{ss.n_clusters}clusters'
    fpath = path.join(ss.args["reports_db_dir"], save_file + '.png')
    plt.savefig(fpath, dpi=ss.args['plot_reports']['dpi'])
    plt.show()

def split_list(d, lst_lens):
    """
    Split a 1D array `d` into sub-arrays of lengths
    specified in the list `lst_lens`. Return a list
    of sub-arrays
    """
    # Calculate the cumulative sum of lengths
    cum_lengths = np.cumsum([0] + lst_lens)

    # Drop the leading 0 and the last item in the cum_lengths list
    subarrays = np.split(d, cum_lengths[1:-1])

    return subarrays

def get_ids_for_physician(ss, lst_hid):
    """
    Return a dictionary physician -> list of indecies
    for elements in 'lst_hid' corresponding to each
    physician.
    """
    if ss.db is None:
        ss.db = ReportsDB(ss.reports_db_pathname)

    lst_phys = ss.args['select_physicians']

    if not lst_phys:
        lst_phys = ss.db.list_physicians()

    phys2hid = {ph:set(ss.db.get_ids_for_physician(ph)) for ph in lst_phys}
    phys2idx = defaultdict(list)

    for i,hid in enumerate(lst_hid):
        for ph in phys2hid:
            if hid in phys2hid[ph]:
                phys2idx[ph].append(i)
                break

    return phys2idx

def subsample_idx(ss, phys2idx, nsamples):
    """
    Randomly choose indecies from the phys2idx dictionary so that
    total number of all indecies is equal to nsamples and return
    'sub-sampled' phys2idx. The algorithm tries to spread nsamples
    equally among physicians. If equal share is larger than total
    number of samples per physician, than all her samples will be
    used.

    NOTE: this function is only called when nsamples is less than
    total number of samples in phys2idx

    Args:
        ss: ref to this app object
        phys2idx(dict): physician -> list if indecies
        nsamples (int): total number of samples required

    Returns:
        new_phys2idx(dict): physician -> 1D array of indecies.
    """
    nphys = len(phys2idx)   # Number of physicians

    if nsamples < nphys:
        raise ValueError('subsample_idx(): number of samples should be >= number of physicians')

    # Initialize the 'phys2n', 'done' dictionaries
    n0 = int(np.floor(nsamples/nphys))  # Tentative # of samples per physician
    nleft = nsamples - n0*nphys         # Unassigned samples
    phys2n = {k:n0 for k in phys2idx}
    done = {k:False for k in phys2idx}

    while sum(done.values()) < nphys:
        for k in phys2n:
            if done[k]:
                continue

            # Adjust where assigned number is too large
            nk = len(phys2idx[k])
            if phys2n[k] >= nk:
                nleft += phys2n[k] - nk
                phys2n[k] = nk
                done[k] = True

        if nleft == 0:
            break       # Nothing else to redistribute

        # Redistribute nleft samples
        n_notdone = nphys - sum(done.values())

        if n_notdone == 0:
            break

        if nleft < n_notdone:
            # Distribute nleft among 1st not done
            for k in phys2n:
                if done[k]:
                    continue

                phys2n[k] += 1

                if phys2n[k] == len(phys2idx[k]):
                    done[k] = True

                nleft -= 1

                if nleft == 0:
                    break
        else:
            nadd = int(np.floor(nleft/n_notdone))   # Samples to add to each not done yet
            nleft -= nadd*n_notdone

            for k in phys2n:
                if done[k]:
                    continue

                phys2n[k] += nadd

    # --- end while() ------------------------

    # Here physicians with done = True have all there idx used;
    # others have phys2n[k] < len(phys2idx[k])
    """
    # DEBUG
    if nsamples <= sum(len(v) for v in phys2idx.values()):
        assert sum(v for v in phys2n.values()) == nsamples

    for k in phys2n:
        if done[k]:
            assert phys2n[k] == len(phys2idx[k])
        else:
            assert phys2n[k] < len(phys2idx[k])
    """

    # Return a sub-sampled copy of phys2idx
    ret = phys2idx.copy()
    for k in phys2idx:
        if done[k]:
            continue

        ret[k] = ss.rng.choice(phys2idx[k], phys2n[k], replace=False)

    return ret

@timeit
def low_dim_embedding(ss, data):
    """
    Embed high dimensional data into low dimensions, using
    Multi Dimensional Scaling (MDS) or t-Distributed Stochastic
    Neighbor Embedding (tSNE).

    Args:
        ss: reference to this app object
        data (ndarray): shape (npoints, n_high_dims) the high dimensional data

    Returns:
        low_dim_data (ndarray): shape (npoints, n_low_dims) the embedded
            low dimensional data
    """
    if ss.args['plot_reports']['low_dim_embedding'] == 'MDS':
        mds = manifold.MDS(random_state = ss.seed, **ss.args['plot_reports']['MDS'])
        low_dim_data = mds.fit_transform(data)      # n_points X n_components
    elif ss.args['plot_reports']['low_dim_embedding'] == 'tSNE':
        tsne = manifold.TSNE(random_state = ss.seed, **ss.args['plot_reports']['tSNE'])
        low_dim_data = tsne.fit_transform(data)
    else:
        raise ValueError('Unrecognized embedding {} requested'.format(\
                ss.args['plot_reports']['low_dim_embedding']))

    return low_dim_data

if __name__ == '__main__': 
    # Unit tests
    class ss:
        pass

    #phys2idx = {'a': [0,1,2,3,4,5,6,7,8,9], 'b': [0,1,2,3,4,5,6,7],
    #        'c': [0,1]}
    phys2idx = {'a': [0,1,2,], 'b': [0,1,2],
            'c': [0,1]}
    #phys2idx = {'a': [0,1,2,3,4,5,6,7,8,9], 'b': [0,1,2,3,4,5,6,7],
    #        'c': [0,1,2,3,4]}
    nsamples = 7

    ss.rng = np.random.default_rng(12345)
    qq = subsample_idx(ss, phys2idx, nsamples)
    print(qq)

