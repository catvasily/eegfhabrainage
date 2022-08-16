[eeg_bpe.py](https://github.com/catvasily/eegfhabrainage/blob/feature/nikolay/eeg_bpe/eeg_bpe.py) contains Python routine for reading and transforming EEG recordings in EDF format, i.e. symbolization of 20 channels EEG timeseries and applying Byte-Pair Encoding algorithm for tokenization of symbolic series.
This tokenized data can be used as an input for classic machine learning models. Demo usage of this code is in [eeg_bpe_run.py](https://github.com/catvasily/eegfhabrainage/blob/feature/nikolay/eeg_bpe/eeg_bpe_run.py)

In order to succesfuly run the code, install needed packages from [requirements.txt](https://github.com/catvasily/eegfhabrainage/blob/feature/nikolay/eeg_bpe/requirements.txt) (works on Python 3.8.10)
```
pip install -r requirements.txt
```

For more details of the pipeline and reasoning behind the approach, please check our [paper](https://github.com/catvasily/eegfhabrainage/blob/feature/nikolay/eeg_bpe/Byte-Pair%20encoding%20for%20EEG_Klymenko.pdf)
