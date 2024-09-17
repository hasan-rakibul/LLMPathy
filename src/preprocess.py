import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

class DataModule:
    def __init__(self, config):

        self.config = config
        
        self.tokeniser = AutoTokenizer.from_pretrained(
                self.config.checkpoint,
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )

        if 'mistral' in self.config.checkpoint or 'llama' in self.config.checkpoint:
            # the tokeniser for mistral and llama does not have the a default pad token
            # so we use eos token as the pad token
            self.tokeniser.pad_token_id = self.tokeniser.eos_token_id
            self.tokeniser.pad_token = self.tokeniser.eos_token
            
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)
    
    def _read_and_process(self, path, send_label, annotation):
        data = pd.read_csv(path, sep='\t')
    
        # keep only the relevant columns
        # if we want to send the label, we need to include the annotation
        if send_label:
            data = data[self.config.data.feature_to_tokenise + self.config.data.demographics + annotation]
        else:
            data = data[self.config.data.feature_to_tokenise + self.config.data.demographics]


        if len(self.config.data.demographics): # if there are demographics
            from sklearn.preprocessing import MinMaxScaler
            data_demog = data[self.config.data.demographics]
            scaler = MinMaxScaler()
            data_demog = pd.DataFrame(
                scaler.fit_transform(data_demog),
                columns=self.config.data.demographics
            )
            # drop the original demographics and replace with the scaled one
            data = data.drop(columns=self.config.data.demographics)
            data = pd.concat([data, data_demog], axis=1)

        return data

    def _tokeniser_fn(self, sentence):
        if len(self.config.data.feature_to_tokenise) == 1: # only one feature
            return self.tokeniser(
                sentence[self.config.data.feature_to_tokenise[0]],
                truncation=True,
                max_length=self.config.data.max_length
            )
        # otherwise tokenise a pair of sentence
        return self.tokeniser(
            sentence[self.config.data.feature_to_tokenise[0]],
            sentence[self.config.data.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.config.data.max_length
        )

    def _process_input(self, data_file, send_label):
        preserve_index = True
        annotation = self.config.data.train_label_list
        data = self._read_and_process(path=data_file, send_label=send_label, annotation=annotation)
        data = Dataset.from_pandas(data, preserve_index=preserve_index) # convert to huggingface dataset
        data = data.map(self._tokeniser_fn, batched=True, remove_columns=self.config.data.feature_to_tokenise) # tokenise
        data = data.rename_column(annotation[0], 'labels')
        data.set_format('torch')
        return data 

    def get_huggingface_data(self, data_file, send_label):
        data = self._process_input(data_file=data_file, send_label=send_label)
        return data
    
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    
    def get_dataloader(self, data_file, send_label, shuffle):
        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.config.seed)

        data = self.get_huggingface_data(data_file=data_file, send_label=send_label)
        return DataLoader(
            data,
            batch_size=self.config.train.batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g
        )
    
    def get_dataloaders_WASSA(self):
        train_dataset = self.get_huggingface_data(data_file=self.config.data.train_file, send_label=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            sampler=torch.utils.data.sampler.RandomSampler(train_dataset),
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers
        )

        val_dataset = self.get_huggingface_data(data_file=self.config.data.val_file, send_label=True)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            sampler=torch.utils.data.sampler.SequentialSampler(val_dataset),
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers
        )

        return train_dataloader, val_dataloader


class SSLDataModule(DataModule):
    def __init__(self, config):
        super().__init__(config)

    def _process_input(self, data_file, send_label):
        data = self._read_and_process(path=data_file, send_label=send_label, annotation=['crowdsourced_empathy', 'gpt_empathy'])
        matched_filter = np.abs(data["crowdsourced_empathy"] - data["gpt_empathy"]) < self.config.data.ssl_threshold
        data_labelled = data[matched_filter]
        data_unlabelled = data[~matched_filter]

        data_labelled["labels"] = data_labelled[["crowdsourced_empathy", "gpt_empathy"]].mean(axis=1)
        data_labelled.drop(columns=["crowdsourced_empathy", "gpt_empathy"], inplace=True)

        # saving labelled data
        good_samples = data_labelled.rename(columns={"labels": "crowdsourced_empathy"}) # rename to match the original column name
        good_samples.to_csv(data_file.replace(".tsv", f"_labelled_{self.config.data.ssl_threshold}.tsv"), sep='\t', index=False)

        data_unlabelled["labels"] = data_unlabelled[["crowdsourced_empathy"]] # why do we have labels in unlabelled data?
        data_unlabelled.drop(columns=["gpt_empathy"], inplace=True)

        data_labelled = Dataset.from_pandas(data_labelled, preserve_index=False)
        data_unlabelled = Dataset.from_pandas(data_unlabelled, preserve_index=False)
        
        data_labelled = data_labelled.map(self._tokeniser_fn, batched=True, remove_columns=self.config.data.feature_to_tokenise)
        data_unlabelled = data_unlabelled.map(self._tokeniser_fn, batched=True, remove_columns=self.config.data.feature_to_tokenise)

        data_labelled.set_format('torch')
        data_unlabelled.set_format('torch')

        return data_labelled, data_unlabelled
    
    def get_huggingface_data(self, data_file, send_label):
        data_labelled, data_unlabelled = self._process_input(data_file=data_file, send_label=send_label)
        return data_labelled, data_unlabelled
    
    def get_dataloader(self, data_file, send_label, shuffle):
        g = torch.Generator()
        g.manual_seed(self.config.seed)

        data_labelled, data_unlabelled = self.get_huggingface_data(data_file=data_file, send_label=send_label)
        
        y_mean = data_labelled["labels"].mean().item()
        y_std = data_labelled["labels"].std().item()

        dataloader_labelled = DataLoader(
            data_labelled, 
            batch_size=self.config.train.batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g
        )

        dataloader_unlabelled = DataLoader(
            data_unlabelled, 
            batch_size=self.config.train.batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g
        )

        return dataloader_labelled, dataloader_unlabelled, y_mean, y_std

class DataModuleFromRaw:
    def __init__(self, config):

        self.config = config
        
        self.tokeniser = AutoTokenizer.from_pretrained(
                self.config.checkpoint,
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )

        if 'mistral' in self.config.checkpoint or 'llama' in self.config.checkpoint:
            # the tokeniser for mistral and llama does not have the a default pad token
            # so we use eos token as the pad token
            self.tokeniser.pad_token_id = self.tokeniser.eos_token_id
            self.tokeniser.pad_token = self.tokeniser.eos_token
            
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)
    
    def _raw_to_processed(self, path, send_label):
        print(f"\nReading data from {path}")
        data = pd.read_csv(path, sep='\t', na_values="unknown") # some column includes "unknown"
        print(f"Read {len(data)} samples from {path}")

        # keep revent columns only
        columns_to_keep = self.config.data.feature_to_tokenise + \
            self.config.data.demographics + \
            self.config.data.extra_columns_to_keep

        if send_label:
            columns_to_keep.append(self.config.data.label_column)
        
        data = data[columns_to_keep]

        print("Columns with NaN values:", data.columns[data.isna().any()].tolist())
        data = data.dropna() # drop NaN values
        print(f"Removed NaN values if existed. {len(data)} samples remaining.\n")

        assert data.isna().any().any() == False, "There are still NaN values in the data."
        assert data.isnull().any().any() == False, "The are still null values in the data"
        
        return data.copy() # return a copy to avoid modifying the original data

    def _tokeniser_fn(self, sentence):
        if len(self.config.data.feature_to_tokenise) == 1: # only one feature
            return self.tokeniser(
                sentence[self.config.data.feature_to_tokenise[0]],
                truncation=True,
                max_length=self.config.data.max_length
            )
        # otherwise tokenise a pair of sentence
        return self.tokeniser(
            sentence[self.config.data.feature_to_tokenise[0]],
            sentence[self.config.data.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.config.data.max_length
        )

    def get_hf_data(self, data_path_list, send_label):
        # we may combine the data from different versions
        for data_path in data_path_list:
            data = self._raw_to_processed(data_path, send_label)
            if 'all_data' in locals():
                all_data = pd.concat([all_data, data])
            else:
                all_data = data

        print(f"Total number of samples: {len(all_data)}\n")
        
        # add sample_id column
        all_data['sample_id'] = range(len(all_data))        
        all_data_hf = Dataset.from_pandas(all_data, preserve_index=False) # convert to huggingface dataset
        
        all_data_hf = all_data_hf.map(self._tokeniser_fn, batched=True, remove_columns=self.config.data.feature_to_tokenise) # tokenise
        all_data_hf = all_data_hf.rename_column(self.config.data.label_column, 'labels')
        all_data_hf.set_format('torch')
        return all_data_hf
    
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    
    def get_dataloader(self, data_path_list, send_label, shuffle):
        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.config.seed)

        hf_data = self.get_hf_data(data_path_list=data_path_list, send_label=send_label)
        return DataLoader(
            hf_data,
            batch_size=self.config.train.batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g
        )
    