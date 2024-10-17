import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import logging
from sklearn.preprocessing import MinMaxScaler

from utils import log_info

logger = logging.getLogger(__name__)

class DataModuleFromRaw:
    def __init__(self, config):

        self.config = config
        
        self.tokeniser = AutoTokenizer.from_pretrained(
                self.config.plm,
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )
            
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)
    
    def _raw_to_processed(self, path, have_label, val):
        log_info(logger, f"\nReading data from {path}")
        data = pd.read_csv(path, sep='\t', na_values="unknown") # some column includes "unknown"
        log_info(logger, f"Read {len(data)} samples from {path}")

        # "2024" has different column names
        if "2024" in path:
            raise NotImplementedError("2024 data is not supported yet.")

        # keep revent columns only
        columns_to_keep = self.config.feature_to_tokenise + \
            self.config.extra_columns_to_keep

        # if it is val of 2022 and 2023, the labels are separate files
        if val:
            try:
                goldstandard = pd.read_csv(
                    self.config.val_goldstandard_file, 
                    sep='\t',
                    header=None # had no header in the file
                )
                # first column is empathy
                goldstandard = goldstandard.rename(columns={0: self.config.label_column})
                data = pd.concat([data, goldstandard], axis=1)
            except:
                log_info(logger, "No goldstandard file is read. Assuming it is not required.")

        if have_label:
            columns_to_keep.append(self.config.label_column)
            if not val: # means it is train
                columns_to_keep.extend(self.config.extra_columns_to_keep_train) # this is a list
        
        selected_data = data[columns_to_keep]

        if len(self.config.demographics) > 0:
            numeric_data = data[self.config.demographics]
            scaler = MinMaxScaler()
            numeric_data = pd.DataFrame(
                scaler.fit_transform(numeric_data),
                columns=self.config.demographics
            )
            selected_data = pd.concat([selected_data, numeric_data], axis=1)
 
        log_info(logger, f"Columns with NaN values: {selected_data.columns[selected_data.isna().any()].tolist()}")
        selected_data = selected_data.dropna() # drop NaN values
        log_info(logger, f"Removed NaN values if existed. {len(selected_data)} samples remaining.\n")

        assert selected_data.isna().any().any() == False, "There are still NaN values in the data."
        assert selected_data.isnull().any().any() == False, "The are still null values in the data"

       
        return selected_data.copy() # return a copy to avoid modifying the original data

    def _tokeniser_fn(self, sentence):
        if len(self.config.feature_to_tokenise) == 1: # only one feature
            return self.tokeniser(
                sentence[self.config.feature_to_tokenise[0]],
                truncation=True,
                max_length=self.config.max_length
            )
        # otherwise tokenise a pair of sentence
        return self.tokeniser(
            sentence[self.config.feature_to_tokenise[0]],
            sentence[self.config.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.config.max_length
        )

    def get_hf_data(self, data_path_list, have_label, val):
        # we may combine the data from different versions
        for data_path in data_path_list:
            data = self._raw_to_processed(data_path, have_label, val)
            if 'all_data' in locals():
                all_data = pd.concat([all_data, data])
            else:
                all_data = data

        log_info(logger, f"Total number of samples: {len(all_data)}\n")
        
        # add sample_id column
        all_data['sample_id'] = range(len(all_data))     
        all_data_hf = Dataset.from_pandas(all_data, preserve_index=False) # convert to huggingface dataset
        
        # tokenise
        all_data_hf = all_data_hf.map(
            self._tokeniser_fn, 
            batched=True,
            remove_columns=self.config.feature_to_tokenise
        )
        if have_label:
            all_data_hf = all_data_hf.rename_column(self.config.label_column, 'labels')
        all_data_hf.set_format('torch')
        return all_data_hf
    
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    
    def _get_dl(self, data_path_list, have_label, shuffle, val=False):
        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.config.seed)

        hf_data = self.get_hf_data(data_path_list=data_path_list, have_label=have_label, val=val)
        return DataLoader(
            hf_data,
            batch_size=self.config.batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g
        )
    def get_train_dl(self, data_path_list):
        return self._get_dl(data_path_list, have_label=True, shuffle=True)
    
    def get_val_dl(self, data_path_list):
        # depending on data_name, the labels can be in different file
        return self._get_dl(data_path_list, have_label=True, shuffle=False, val=True)
    
    def get_test_dl(self, data_path_list):
        return self._get_dl(data_path_list, have_label=False, shuffle=False) # we don't have labels
    