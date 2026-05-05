"""
Top level script for running EEG-based classification steps.

To add a new step to this script:

1. Encapsulate the step logic in a function, for example::

       def my_step(ss):
           ...

2. Add an entry to the ``cases`` dictionary in the EPILOGUE section::

       'my_step': my_step

Steps may reside in separate Python files. In that case add the corresponding
import statements to this file.

All input parameters are read from the JSON file referenced by
``INPUT_JSON_FILE``::

    INPUT_JSON_FILE = '<my-file>.json'

The content of that JSON file is available as ``ss.args`` in each step.

Variables shared between steps should be stored as attributes on ``ss``, for
example::

    ss.common_var = common_var_value

This can be done in ``__init__(ss)`` together with other project-specific
initialization.

The sequence of steps to execute should be set via ``to_run`` in the JSON,
for example::

    to_run = ("init", "step1", "step6")

For batch mode with different parameter sets per run:

* Add ``batch_run`` to ``INPUT_JSON_FILE`` and set it to ``true``.

* Add ``batch_job_parms`` as a list of dictionaries such as
  ``[{"ijob": <job-num>, <key>: <value>, ...}]`` where each key/value pair
  overrides one JSON setting for the given ``ijob``.

* With no CLI arguments, all configurations in ``batch_job_parms`` are run.

* With CLI arguments and batch mode enabled, the first CLI argument is treated
  as the ``ijob`` identifier and only that configuration is run.

Available steps:

* ``xgboost``: classify CWT amp distributions using XGBoost.
* ``summarize``: collect results from available ``.pkl`` files and create a summary.
* ``summary_plots``: create summary bar plots from ``.csv`` output.
* ``feature_importance``: show feature importances for a specified label.
* ``predict``: use a trained model in prediction mode.
"""

import sys
import os
import os.path as path
import socket
import copy
import commentjson as cjson
from pathlib import Path

from do_xgboost import do_xgboost
from do_cls_summarize import do_cls_summarize
from cls_feature_importance import cls_feature_importance
from cls_predict import cls_predict
from plot_cls_summary import plot_cls_summary

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)  # full pathname to source file

#sys.path.append(path.dirname(path.dirname(__file__))+ "/beam-python")

#-------------------------------------
INPUT_JSON_FILE = "classifier_input.json"      # This script's input parameters
#-------------------------------------

def init(ss):
    """
    Perform project specific initializations and add global attributes
    that may be reused in different steps to the app object instance.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    # This following call will set:
    # ss.host, ss.cluster_job, and paths to various folders
    # depending on the host
    setup_paths(ss)

    if ss.cluster_job:
        # One can allow making plots. Only showing them should
        # be disabled (just to avoid warnings - they will be disabled anyways)
        ss.args['show_plots'] = False    

def setup_paths(ss):
    """
    Generate various fully qualified file and directory names.
    This function is called from the main application class constructor.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    args = ss.args
    ss.host, ss.cluster_job = get_host(ss)

    data_root = Path(ss.args['hosts'][ss.host][ss.args['what']]['data_root'])
    
    ss.data_dir = lambda hospital: data_root / hospital

    ss.tfd_pname = lambda hospital, scan_id:   \
            ss.data_dir(hospital) / (scan_id + \
            ('_tfd.hdf5' if ss.args['what'] == 'sensors' else  '_src_tfd.hdf5'))

    ss.db_root = Path(ss.args['hosts'][ss.host]['db_root'])
    ss.eeg_info_db_pname = ss.db_root / ss.args['eeg_info_db']
    ss.classifications_db_pname = ss.db_root / ss.args['classifications_db']

    ss.out_root = Path(ss.args['hosts'][ss.host][ss.args['what']]['out_root'])

    # Utility to create a string from the hosptialt list like this:
    # ['Burnaby', 'Abbotsford'] -> 'BA'
    ss.hlist = lambda hospital_list: ''.join(sorted(s[0] for s in hospital_list))  

    def _reducer_pickle_tag():      # A helper to construct reducer str for pickle name
        if bool(ss.args.get('use_consensus_cv', False)):
            return 'ccv'
        method = str(ss.args.get('dim_reduction', '')).lower()

        if method == 'pca':
            return 'pca'
        if method == 'epi_features':
            return 'epi'
        return ''  # e.g. to_lobes -> no reducer tag

    def _batch_suffix():
        return f'_ijob{ss.ijob}' if ss.batch_run else ''

    ss.cls_pkl_pname = lambda hlist, label, nparms, standardize, ignore_confidence: \
            ss.out_root / (
                f'xgb_{hlist}_{label}_{_reducer_pickle_tag()}_nparms{nparms}_std{standardize}_ignoreConf{ignore_confidence}{_batch_suffix()}.pkl'
                if _reducer_pickle_tag()
                else f'xgb_{hlist}_{label}_nparms{nparms}_std{standardize}_ignoreConf{ignore_confidence}{_batch_suffix()}.pkl'
            )

    def _prcrv_from_pkl(pkl_pname):
        stem = pkl_pname.stem
        if stem.startswith('xgb_'):
            stem = f'prcrv_{stem[4:]}'
        else:
            stem = f'prcrv_{stem}'
        return pkl_pname.with_name(f'{stem}.png')

    ss.prcrv_png_pname = lambda hlist, label, nparms, standardize, ignore_confidence: \
            _prcrv_from_pkl(ss.cls_pkl_pname(hlist, label, nparms, standardize, ignore_confidence))

def get_host(ss):
    """
    Detect the host we are running at; return its name and
    the `cluster_job` flag.
    """
    host_found = False
    host = socket.getfqdn()

    for key in ss.args['hosts']:
        if key in host:
            host = key
            host_found = True
            break

    if not host_found:
        raise ValueError(f'Host is not listed in the {INPUT_JSON_FILE}.')

    return host, ss.args['hosts'][host]['cluster_job']

# --------------------------------------------------------
#                    EPILOGUE                             
# --------------------------------------------------------
class _app:
    # ------
    # Cases: add your steps here in the form "my_step":my_step,
    # ------
    cases = {
        # Steps to run go here:
        'xgboost': do_xgboost,
        'summarize': do_cls_summarize,
        'summary_plots': plot_cls_summary,
        'feature_importance': cls_feature_importance,
        'predict': cls_predict,
    }

    def __init__(ss):
        """
        Read all input parameters from JSON file and prepare for running
        the individual steps.

        Args:
            ss(obj): reference to this app object

        Returns:
            Nothing

        """
        with open(_pathname(INPUT_JSON_FILE), 'r') as fp:
            ss.args = cjson.loads(fp.read())

        ss.base_args = copy.deepcopy(ss.args)   # Save the original configuration
                                                # in case batch job will be run

        ss.batch_run = bool(ss.base_args.get('batch_run', False))

        ss.cli_arg_provided = len(sys.argv) > 1

        if not ss.cli_arg_provided:   # No command line args
            ss.cli_ijob = 0
        else:
            ss.cli_ijob = int(sys.argv[1])

        ss.ijob = ss.cli_ijob       # Save the batch job number. It will be 0
                                    # if not running in batch mode

    def __call__(ss, name, *args, **kwargs):
        """
        Execute requested step(s). This function allows to use code
        ```
        ss(name)
        ```
        to execute step specified by `name`.

        Args:
            ss(obj): reference to this app object
            name(str): step name
            args(list): step positioning args (if any)
            kwargs(dict): step key-word args (if any)
            
        Returns:
            Nothing

        """
        not_found = True

        for f in ss.cases:
            if f == name:
                ss.cases[f](ss, *args, **kwargs)
                not_found = False
                break

        if not_found:
            raise ValueError(f'Requested method "{name}" not found')

    def run_steps(ss):
        """Run all steps listed in args['to_run']."""
        for name in ss.args['to_run']:
            ss(name)

    # ----------------------------------------------------------
    # The following routines are needed to execute in batch mode.
    # ----------------------------------------------------------
    def merge_dict_overrides(ss, target, overrides, context=''):
        """
        Recursively update existing keys in target dictionary.

        Args:
            target(dict): dictionary to update
            overrides(dict): values to apply
            context(str): nested path for error messages

        Returns:
            Nothing
        """
        for key, value in overrides.items():
            full_key = f'{context}.{key}' if context else key   # note that bool('') = False

            if key not in target:
                raise ValueError(f'Batch parameter "{full_key}" is not a valid key in {INPUT_JSON_FILE}.')

            if isinstance(value, dict):
                if not isinstance(target[key], dict):
                    raise ValueError(
                        f'Batch parameter "{full_key}" is a dictionary, but base value is not a dictionary.'
                    )

                # Recurse inside the nested key
                ss.merge_dict_overrides(target[key], value, full_key)
            else:
                # Value is not a dict just set it
                target[key] = value


    def apply_batch_job_args(ss, job_args):
        """
        Apply one batch job dictionary to current app args.

        Args:
            job_args(dict): one element from the `batch_job_parms` list

        Returns:
            Nothing
        """
        for key, value in job_args.items():
            if key == 'ijob':
                continue

            if key not in ss.args:
                raise ValueError(f'Batch parameter "{key}" is not a valid key in {INPUT_JSON_FILE}.')

            if isinstance(value, dict):
                if not isinstance(ss.args[key], dict):
                    raise ValueError(
                        f'Batch parameter "{key}" is a dictionary, but base value is not a dictionary.'
                    )

                ss.merge_dict_overrides(ss.args[key], value, key)
            else:
                ss.args[key] = value

    def run_single_job(ss):
        """Run the script once using base parameters and optional CLI ijob."""
        ss.args = copy.deepcopy(ss.base_args)
        ss.ijob = ss.cli_ijob
        init(ss)
        ss.run_steps()

    def run_batch_jobs(ss):
        """
        Run all jobs from batch_job_parms, applying per-job parameter overrides.
        """
        jobs = ss.base_args.get('batch_job_parms', None)

        if not isinstance(jobs, list):
            raise ValueError('Batch mode requires key "batch_job_parms" to be a list of dictionaries.')

        if ss.cli_arg_provided:
            # We are hear if some command line args are available. In this case
            # ss.cli_ijob is equal to int(arg #1)
            matching_jobs = [job_args for job_args in jobs
                             if isinstance(job_args, dict) and job_args.get('ijob', None) == ss.cli_ijob]

            if len(matching_jobs) == 0:
                # Fail if requested job number is not listed among ijob keys of specified
                # batch job parms
                available_ijobs = [job_args.get('ijob', None) for job_args in jobs if isinstance(job_args, dict)]
                raise ValueError(
                    f'No batch job found with ijob={ss.cli_ijob}. '
                    f'Available ijob values: {available_ijobs}'
                )

            if len(matching_jobs) > 1:
                # Fail for ambiguous setting: two or more batch jobs have the same
                # ijob key value
                raise ValueError(
                    f'More than one batch job found with ijob={ss.cli_ijob}. '
                    'Each batch job must have a unique ijob.'
                )

            jobs_to_run = [(ss.cli_ijob, matching_jobs[0])]
        else:
            # When no CLI args provided but batch mode requested -
            # run all listed jobs sequentially one after another
            jobs_to_run = list(enumerate(jobs))

        # Now jobs_to_run is a list of tuples: [(ijob, job_parms_dict)]
        for i, job_args in jobs_to_run:
            if not isinstance(job_args, dict):
                raise ValueError(f'Batch job at index {i} must be a dictionary.')

            if 'ijob' not in job_args:
                raise ValueError(f'Batch job at index {i} must contain key "ijob".')

            ss.args = copy.deepcopy(ss.base_args)
            ss.ijob = job_args['ijob']
            ss.apply_batch_job_args(job_args)
            init(ss)
            ss.run_steps()

if __name__ == '__main__': 
    # Uncomment this code if you need calls like this:
    #   _app.<case_func>()
    # for some reason - that is, if
    #   _app(<step>)
    # does not work for you.
    # for c in _app.cases:
    #     setattr(_app, c, _app.cases[c])
      
    this_app = _app()

    if this_app.batch_run:
        this_app.run_batch_jobs()
    else:
        this_app.run_single_job()

# -------------- end of Epilogue --------------------------



