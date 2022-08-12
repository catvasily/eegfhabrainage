import os
import pandas as pd
import numpy as np
import mne
mne.set_log_level('warning')
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

class Collector:
    
    def __init__(self, folder_path):
        files = os.listdir(folder_path)
        self.edf_files = [folder_path + file for file in files]
        
        self.scan_ids = []
        self.annotations = []
        
    def collect_annotations(self):
        
        for edf_path in self.edf_files:
  
            try:
                raw = mne.io.read_raw_edf(edf_path, verbose='warning', preload=False)
                annots = raw.annotations.description

                self.scan_ids.append(edf_path.split('/')[-1].split('.')[0])
                self.annotations.append(annots)
            except:
                print('read failed')
                
        self.annotations_df = pd.DataFrame()
        self.annotations_df['scan_id'] = self.scan_ids
        self.annotations_df['annotations'] = self.annotations
        
        return self.annotations_df
    
    
    def count_annotations(self, size=None):
        
        full = np.empty(0)      
        size = self.annotations_df['annotations'].shape[0]

        for i in range(size):
            array = self.annotations_df['annotations'][i]
            full = np.concatenate((full, array))
        
        counts = Counter(full)
        df = pd.DataFrame.from_dict(counts, orient='index')
        df = df.reset_index()
        df.columns = ['annot', 'count']
        self.count_df = df.sort_values(by='count', axis=0, ascending=False)
        
    def lookup_annotation(self, query):
        
        return self.count_df[self.count_df['annot'].str.contains(query)]