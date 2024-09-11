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
        data = self._read_and_process(path=data_file, send_label=send_label, annotation=['crowdsourced_empathy', 'gpt_empathy'])
        data = Dataset.from_pandas(data, preserve_index=preserve_index) # convert to huggingface dataset
        data = data.map(self._tokeniser_fn, batched=True, remove_columns=self.config.data.feature_to_tokenise) # tokenise
        data = data.rename_column('crowdsourced_empathy', 'labels')
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
