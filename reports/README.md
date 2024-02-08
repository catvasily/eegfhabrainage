# EEG clinical reports processing

## Summary
Code in this folder consists of two major parts.

1. The first part focuses on **parsing the original clinical reports data**. The latter comes as an ASCII file formatted in a special way, and encompassing approximately 45,500 EEG reports from 4 hospitals and 18 neurologists. During the processing of the original reports, all personal identifiable information (PII) is removed. Also, new report IDs (the *Hashed IDs*) are generated and the actual names of neurologists are substituted with aliases. These anonymized reports are then stored in the SQLite database named *`processed_reports.db`*.  
   Given the sensitive nature of information involved in this stage, the corresponding routines are only accessible in the (protected) complete version of this repository.

2. The second part contains code designed for **interacting with the "clean" reports database, conducting NLP analyses of the data** including word- and report-level vector space embeddings, **clustering of data** and **visualizing the embedding and clustering outcomes**.
 
Either part can be executed by running a correpsonding top level script: **`run_reports.py`** for the parsing and anonymization part, or **`process_reports.py`** for NLP processing, clustering and visualization.

Each top level script may perform a sequence of sub-tasks, and is controlled by a corresponding JSON file: **`reports_input.json`** and **`proc_input.json`**, respectively. In particular, the tasks to run are listed under the `"to_run"` key in the JSON file. All input arguments and configuration parameters specific to each sub-task a listed under their dedicated keys in JSON files.

NOTE that both `reports_input.json` and `proc_input.json` have extensive comments inside them, and thus violate the standard vanilla JSON format. To process such JSON files properly one needs to import `commentjson` module and to use routines there in in user's Python code.

All further details regarding available sub-tasks and script parameters can be found in the [auto-generated documentation](../doc/_build/html/index.html).

## Installation
To run the code, one needs to install some machine leaning, PII analyses and NLP tools in corresponding virtual environment, as shown in the code snippet below:
```
python3 -m pip install pip --upgrade
python3 -m pip install presidio_analyzer
python3 -m pip install presidio_anonymizer
python3 -m spacy download en_core_web_lg
python3 -m pip install --upgrade gensim
python3 -m pip install -U scikit-learn

```

