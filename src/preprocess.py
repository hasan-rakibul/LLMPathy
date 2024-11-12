import random
import numpy as np
import pandas as pd
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import logging

from utils import log_info, log_debug, read_file

logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

class DataModuleFromRaw:
    def __init__(self, config):

        self.config = config
        
        self.tokeniser = AutoTokenizer.from_pretrained(
                self.config.plm,
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )
            
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)

    def _label_fix_inplace(self, data: pd.DataFrame) -> None:
        """
        Replace 'empathy' values with 'llm_empathy' if their difference exceeds the threshold 'alpha'.

        in place because pandas dataframes are mutable

        """
        assert self.config.label_column in data.columns, f"{self.config.label_column} column not found in the data"
        assert self.config.llm_column in data.columns, f"{self.config.llm_column} column not found in the data"
        # Calculate the absolute difference between 'empathy' and 'llm_empathy'
        condition = np.abs(data[self.config.label_column] - data[self.config.llm_column]) > self.config.alpha
        log_info(logger, f"Number of labels updated (crowdsourced labels --> LLM labels): {condition.sum()}")
        log_debug(logger, f"Replacing at indices: {data[condition].index.tolist()}")

        # Replace 'empathy' values where the condition is True
        data.loc[condition, self.config.label_column] = data.loc[condition, self.config.llm_column]

        data.drop(columns=[self.config.llm_column], inplace=True)
    
    def _one_hot_encode_demog(self, categorical_data: pd.DataFrame, categorical_features: List, possible_categories: Dict) -> pd.DataFrame:
        assert set(categorical_features) == set(categorical_data.columns), f"Categorical data columns {categorical_data.columns} do not perfectly match with the categorical features {categorical_features}"
        
        categorical_data = categorical_data.astype(int)
        
        encoded_data = pd.get_dummies(categorical_data, columns=categorical_features, prefix=categorical_features)

        # ensure that all possible categories are present in the encoded data
        for col in categorical_features:
            for cat in possible_categories[col]:
                if f"{col}_{cat}" not in encoded_data.columns:
                    encoded_data[f"{col}_{cat}"] = 0

        encoded_data = encoded_data.astype(int)
        return encoded_data
    
    def _raw_to_processed(self, path: str, have_label: bool, mode: str) -> pd.DataFrame:
        log_info(logger, f"\nReading data from {path}")
        data = read_file(path)
        
        log_info(logger, f"Read {len(data)} samples from {path}")

        # keep revent columns only
        columns_to_keep = self.config.feature_to_tokenise + \
            self.config.extra_columns_to_keep

        # if it is val of 2022 and 2023, the labels are separate files
        val_goldstandard_file = None
        if "WASSA23_essay_level_dev" in path:
            val_goldstandard_file = "data/NewsEmp2023/goldstandard_dev.tsv"
        elif "messages_dev_features_ready_for_WS_2022" in path:
            val_goldstandard_file = "data/NewsEmp2022/goldstandard_dev_2022.tsv"
        if val_goldstandard_file is not None:
            assert os.path.exists(val_goldstandard_file), f"File {val_goldstandard_file} does not exist."
            goldstandard = pd.read_csv(
                val_goldstandard_file, 
                sep='\t',
                header=None # had no header in the file
            )
            # first column is empathy
            goldstandard = goldstandard.rename(columns={0: self.config.label_column})
            data = pd.concat([data, goldstandard], axis=1)

        if have_label:
            columns_to_keep.append(self.config.label_column)
        
        if mode == "train":
            columns_to_keep.extend(self.config.extra_columns_to_keep_train) # this is a list
        
        if mode == "train_only_LLM":
            if self.config.label_column in data.columns:
                data.drop(columns=[self.config.label_column], inplace=True) # remove the label column
            data.rename(columns={self.config.llm_column: self.config.label_column}, inplace=True)

        selected_data = data[columns_to_keep]

        
        if self.config.use_demographics:
            log_info(logger, f"Using demographics data.")

            continous_features = ["age", "income"]
            categorical_features = ["gender", "education", "race"]
            demog_columns = continous_features + categorical_features
     
            if "2024" in path: # volatile check, depending on the dir/file name
                # require mapping for 2024 demographics
                # demographics mapping are available from PER dataset
                per_train = pd.read_csv("data/NewsEmp2024/demographic-from-PER-task/trac4_PER_train.csv", index_col=0)
                per_dev = pd.read_csv("data/NewsEmp2024/demographic-from-PER-task/trac4_PER_dev.csv", index_col=0)
                per_test = pd.read_csv("data/NewsEmp2024/demographic-from-PER-task/goldstandard_PER.csv", index_col=None)
                assert per_train.columns.to_list() == per_dev.columns.to_list() == per_test.columns.to_list()
                
                demog_map = pd.concat([per_train, per_dev, per_test], ignore_index=True)
                demog_map.drop_duplicates(subset=['person_id'], inplace=True)
                
                data["person_id"] = data["person_id"].astype(str) # for merging
                demog_map["person_id"] = demog_map["person_id"].astype(str) # for merging

                only_demog_map = demog_map[['person_id'] + demog_columns] # person_id is important for mapping
                data = pd.merge(data, only_demog_map, on='person_id', how='left', validate='many_to_one')

            assert set(demog_columns).issubset(data.columns), f"Some/all demographics columns {demog_columns} not found in the data"
            
            possible_categories = {
                'gender': [1, 2, 5],
                'education': [1, 2, 3, 4, 5, 6, 7],
                'race': [1, 2, 3, 4, 5, 6]
            }

            demog_data = data[demog_columns]
            # handle missing value
            
            if demog_data.isna().any().any():
                log_info(logger, f"Demographic columns {demog_data.columns[demog_data.isna().any()].tolist()} have {demog_data.isna().sum().sum()} NaN values in total.")
                demog_data.fillna(value={col: demog_data[col].mode().values[0] for col in demog_columns}, inplace=True)
                log_info(logger, f"Filled these NaN values with mode of the column.\n")

            categorical_demog = demog_data[categorical_features]
            one_hot_encoded_data = self._one_hot_encode_demog(categorical_demog, categorical_features, possible_categories)

            continous_data = demog_data[continous_features]
            scaler = MinMaxScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(continous_data),
                columns=continous_features
            )

            selected_data = pd.concat([selected_data, one_hot_encoded_data, scaled_data], axis=1)

        if mode == "train" and "alpha" in self.config: # adding alpha in config during the beginning of training
            log_info(logger, f"Updating labels of {path} file.\n")
            self._label_fix_inplace(selected_data)

        if selected_data.isna().any().any(): 
            log_info(logger, f"Columns {selected_data.columns[selected_data.isna().any()].tolist()} have {selected_data.isna().sum().sum()} NaN values in total.")
            selected_data = selected_data.dropna() # drop NaN values; this could be NaN if the essay or label is None, so we drop the whole row
            log_info(logger, f"Removed rows with any NaN values. {len(selected_data)} samples remaining.\n")

        assert selected_data.isna().any().any() == False, "There are still NaN values in the data."
        assert selected_data.isnull().any().any() == False, "The are still null values in the data"

        return selected_data

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

    def get_hf_data(self, data_path_list, have_label, mode):
        # we may combine the data from different versions
        for data_path in data_path_list:
            data = self._raw_to_processed(data_path, have_label, mode)
            if 'all_data' in locals():
                all_data = pd.concat([all_data, data])
            else:
                all_data = data
        
        if mode == "train":
            # add the train_file_only_LLM_list
            for data_path in self.config.train_file_only_LLM_list:
                # assuming no have_label, as we are using LLM labels as the main labels
                data = self._raw_to_processed(data_path, have_label=False, mode="train_only_LLM")
                all_data = pd.concat([all_data, data])

        log_info(logger, f"Total number of {mode} samples: {len(all_data)}\n")
        
        assert all_data.isna().any().any() == False, "There are still NaN values in the data."
        assert all_data.isnull().any().any() == False, "The are still null values in the data"

        # all_data.to_csv(f"tmp/all_{mode}_data.tsv", sep='\t', index=False) # save the data for debugging

        # add sample_id column
        # all_data['sample_id'] = range(len(all_data))     
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
    
    def _get_dl(self, data_path_list, have_label, shuffle, mode):

        if mode == "test" and "seed" not in self.config:
            # typically, seed should not matter in test and we may not have it in the config
            self.config.seed = 0

        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.config.seed)

        hf_data = self.get_hf_data(data_path_list=data_path_list, have_label=have_label, mode=mode)
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
        return self._get_dl(data_path_list, have_label=True, shuffle=True, mode="train")
    
    def get_val_dl(self, data_path_list):
        # depending on data_name, the labels can be in different file
        return self._get_dl(data_path_list, have_label=True, shuffle=False, mode="val")
    
    def get_test_dl(self, data_path_list, have_label=False):
        return self._get_dl(data_path_list, have_label=have_label, shuffle=False, mode="test") # we have labels in 2024 data
    
