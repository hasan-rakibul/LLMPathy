import os
import numpy as np
import pandas as pd
import torch
import logging
from omegaconf import OmegaConf
import lightning as L
import transformers

from utils import log_info, get_trainer, resolve_logging_dir
from preprocess import DataModuleFromRaw
from model import LightningPLM

logger = logging.getLogger(__name__)

def _calculate_error(model, train_dl):
    # basic pytorch as Lightning predictions become distributed across gpus
    device = model.device
    model.eval()
    
    error_list = []
    sample_ids_list = []
    with torch.no_grad():
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = preds.cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            error = (preds - labels) ** 2
            error_list.extend(error) # array can be appended to a list, it becomes a normal list
            sample_ids_list.extend(batch["sample_id"].cpu().tolist())

    return error_list, sample_ids_list

def _find_noisy_samples_agentic(config, train_dl, val_dl):
    models = [LightningPLM(config) for _ in range(config.num_agents)]
    error_df = pd.DataFrame(index=train_dl.dataset["sample_id"].cpu().tolist())

    raw_logging_dir = config.logging_dir # so that we don't nest
    for i, model in enumerate(models):
        # customise logging_dir per agent
        config.logging_dir = os.path.join(raw_logging_dir, f"agent_{i}")
        trainer = get_trainer(config)
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        
        model = LightningPLM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        error, sample_id = _calculate_error(model, train_dl)
        error_df.loc[sample_id, f"error_{i}"] = error
        
    threshold = error_df.quantile(q=config.noise_level)
    log_info(logger, f"Threshold:\n{threshold}")

    threshold_matrix = error_df < threshold # will be True if error < threshold for each sample

    # keep only the samples that are True for all columns;
    # False in any column will result in NaN, and thus the entire row is dropped
    hc_sample_ids = error_df[threshold_matrix].dropna().index.to_numpy()
    lc_sample_ids = error_df[~threshold_matrix].dropna().index.to_numpy()

    # if there are any samples that are not classified as HC or LC, they will be classified as MC
    mc_sample_ids = error_df.index.difference(hc_sample_ids).difference(lc_sample_ids).to_numpy()
    
    config.logging_dir = raw_logging_dir

    return hc_sample_ids, mc_sample_ids, lc_sample_ids

def _agentic_noise_removal(config):
    datamodule = DataModuleFromRaw(config)
    train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)
    
    hc_sample_ids, mc_sample_ids, lc_sample_ids = _find_noisy_samples_agentic(config, train_dl, val_dl)
    
    log_info(logger, f"HC: {len(hc_sample_ids)}, MC: {len(mc_sample_ids)}, LC: {len(lc_sample_ids)}")
    
    if config.save_agentics_to_disk:
        # Saving noise_indices for analysis...
        np.save(os.path.join(config.logging_dir, "hc_sample_ids_" + str(config.noise_level) + ".npy"), hc_sample_ids)
        np.save(os.path.join(config.logging_dir, "mc_sample_ids_" + str(config.noise_level) + ".npy"), mc_sample_ids)
        np.save(os.path.join(config.logging_dir, "lc_sample_ids_" + str(config.noise_level) + ".npy"), lc_sample_ids)
    
    ####### for mc_set or lc_set, update the label to llm_empathy
    # Convert indices to sets for quick lookup
    mc_set = set(mc_sample_ids)
    lc_set = set(lc_sample_ids)
    
    # If sample_id is in mc_set or lc_set, update the label to llm_empathy
    for batch in train_dl:
        for i, sample_id in enumerate(batch["sample_id"]):
            sample_id = sample_id.item() # index tensor to scalar
            if sample_id in mc_set or sample_id in lc_set:
                batch["labels"][i] = batch[config.llm_column][i]

    log_info(logger, f"Updated labels for {len(mc_set) + len(lc_set)} samples")

    if config.save_agentics_to_disk:
        # save updated train_dl
        torch.save(train_dl, os.path.join(config.logging_dir, "updated_train_dl.pt"))
        log_info(logger, f"Saved updated train_dl to {config.logging_dir}/updated_train_dl.pt")

    return train_dl

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config = OmegaConf.load("config/config_train.yaml")
    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir

    L.seed_everything(config.seed)
    _ = _agentic_noise_removal(config)
