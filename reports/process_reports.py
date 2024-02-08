"""
**A top level script for perfoming various tasks with a clean
(anonymized) EEG reports database**.

To add a step to be run to this script:

1. Encapsulate all related code in a dedicated function  

    `def my_step(ss):  
    ...  
    # NOTE: Step return value (if any) is not used`

2. Add corresponding entry to the cases dictionary in
   the **EPILOGE** section at the bottom of this .py file in the form  

   `'my_step': my_step,`

Steps may reside in separate Python files. In that case
corresponding *import* statements should be added here.
The variables that are intended to be shared between
steps should be defined as attributes of the ss object,
as follows:

   `ss.common_var = common_var_value`

The sequence of steps to be executed is listed under **"to_run"**
key in the JSON input file, for example

   `"to_run": ["init", "step1", "step6"]`

--------------------------------------------------

Available steps:

`'input'`: As is: set all the input and configuration
parameters here. This step should always be run first.

`'word_stats'`: collect words from all reports and run some
stats on those.

`'word2vec'`: construct vector embedding for words in reports.

`'show_similar_words'`: display words similar to given, using word2vec results.

`'doc2vec'`: construct vector embedding for words in reports.

`'show_similar_reports'`: as is.

`'cluster_reports'`: as is.

`'plot_reports'`: plot document vectors embedded into 2D or 3D space.

`'clustering_info_to_reports_db'`: update processed reports DB with clustering
results.

"""
import sys
import os.path as path
import commentjson as cjson
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from reports_db import ReportsDB
from word2vec import word2vec, show_similar_words, doc2vec, show_similar_reports,\
        cluster_reports, plot_reports, clustering_info_to_reports_db

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

INPUT_JSON_FILE = "proc_input.json"    # This script input parameters

def inputs(ss):
    """
    Set up here all the common inputs required by subsequent steps.

    Args:
        ss: This app object

    Returns:
        None
    """
    ss.reports_db_pathname = path.join(ss.args["reports_db_dir"],  ss.args["reports_db_name"])
    ss.db = None        # Database not opened yet
    ss.data = None      # No reports data loaded. Will be set after a call to
                        # word2vec.get_doc_vectors_for_physicians()

    ss.seed = ss.args['seed']
    ss.rng = np.random.default_rng(ss.seed)

def word_stats(ss):
    """
    Collect words from all reports in DB and do some stats.
    Save results to `.csv` file and and also to `ss.df_words` attribute
    in the form of a DataFrame.
    """
    csv_fname = lambda: path.join(ss.args['reports_db_dir'],
            ss.reports_for + '_words.csv')

    if ss.db is None:
        ss.db = ReportsDB(ss.reports_db_pathname)

    reports = get_selected_reports(ss)
    words = Counter()

    for rep in reports:
        words.update(get_words(ss, rep))

    words = words.most_common()
    ss.df_words = pd.DataFrame.from_records(words, columns = ['Word', 'Count'])

    # Save data to .CSV file
    ss.df_words.to_csv(path_or_buf=csv_fname(), sep=',', header=True, index=False)

    # Print all words in order of decreasing count
    print(ss.df_words)

def get_selected_tagged_reports(ss):
    """
    Returns a **set** of tuples `('hash_id', report)` for reports specified in
    the `INPUT_JSON_FILE` file.
    This function also sets `reports_for` attribute of the `ss` object which determines
    the target subset of reports: 'all', or '[<physicians>]', or '[<hospitals>]'.

    """
    join_names = lambda lst: ''.join([w+'_' for w in lst])[:-1]

    reports = set()

    # NOTE: selected physicians have the highest priority;
    # hospitals will only be looked at only if physicians
    # are not specified
    if not ss.args['select_physicians']:
        if not ss.args['select_hospitals']:
            # Neither physicians nor hospitals specified:
            # - process ALL reports
            reports.update(ss.db.get_all_reps().items())
            ss.reports_for = 'all' 
        else:
            # Get reports for selected hospitals
            for h in ss.args['select_hospitals']:
                reports.update(ss.db.get_reps_for_hospital(h).items())

            ss.reports_for = join_names(ss.args['select_hospitals']) 
    else:
        # Get reports for selected phycisians
        for ph in ss.args['select_physicians']:
            reports.update(ss.db.get_reps_for_physician(ph).items())

        ss.reports_for = join_names(ss.args['select_physicians']) 

    return reports

def get_selected_reports(ss):
    """
    Returns a **set** of reports specified in the `INPUT_JSON_FILE` file.
    This function also sets `reports_for` attribute of the `ss` object which determines
    the target subset of reports: 'all', or '[<physicians>]', or '[<hospitals>]'.

    """
    reps = get_selected_tagged_reports(ss)
    return {item[1] for item in reps}

def get_words(ss, text, ignore = None):
    """
    Return a **set** of words from a given text. The returned set is always
    in lower case, and certain types of tokens are ignored.

    Specifically, tokens which are not entirely alphabetic (like 11:30) will
    be dropped. If requested in the JSON config file, the stop words will
    also be dropped. Tokens specified in the `ignore` collection will
    be dropped as well.

    Args:
        ss: reference to this app object
        text(str): the text to be parsed
        ignore(sequence): a collection of words/tokens to be excluded from the
            output

    Returns:
        words(set): a set of words from the `text`, in lower case
    """
    text = re.sub(r'[^a-z\s]', '', text.lower(), flags=re.UNICODE)
    words = text.split()

    if ss.args['ignore_stop_words']:
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]

    if ignore is None:
        return words

    else:
        return [w for w in words if w not in ignore]

# --------------------------------------------------------
#                    EPILOGUE                             
# --------------------------------------------------------
class _app:
    # ------
    # Cases: add your steps here in the form "my_step":my_step,
    # ------
    cases = {
        # Steps to run go here:
        'inputs': inputs,
        'word_stats': word_stats,
        'word2vec': word2vec,
        'show_similar_words': show_similar_words,
        'doc2vec': doc2vec,
        'show_similar_reports': show_similar_reports,
        'cluster_reports': cluster_reports,
        'plot_reports': plot_reports,
        'clustering_info_to_reports_db': clustering_info_to_reports_db,
    }

    def __call__(self, name, *args, **kwargs):
        not_found = True

        for f in self.cases:
            if f == name:
                self.cases[f](self, *args, **kwargs)
                not_found = False
                break

        if not_found:
            raise ValueError(f'Requested method "{name}" not found')

if __name__ == '__main__': 
    for c in _app.cases:
        setattr(_app, c, _app.cases[c])

    this_app = _app()

    with open(pathname(INPUT_JSON_FILE), 'r') as fp:
        this_app.args = cjson.loads(fp.read())

    for name in this_app.args['to_run']:     
        this_app(name)

# -------------- end of Epilogue --------------------------
