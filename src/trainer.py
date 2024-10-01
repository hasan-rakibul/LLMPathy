import os
import logging
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

from utils import log_info
from preprocess import DataModuleFromRaw
from model import LightningPLM

logger = logging.getLogger(__name__)

def _get_trainer(config, devices="auto"):
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        default_root_dir=config.logging_dir,
        deterministic=True,
        logger=True,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min"
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                min_delta=0.01
            )
        ],
        devices=devices
    )

    return trainer


def vanilla_plm(config, train_dl=None):
    datamodule = DataModuleFromRaw(config)

    if not config.test_mode:
        trainer = _get_trainer(config)
        
        if train_dl is None:
            train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)

        val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        with trainer.init_module():
            # model created here directly goes to GPU
            model = LightningPLM(config)

        if config.load_from_checkpoint:
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                ckpt_path=config.load_from_checkpoint
            )
        else:
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
            )

        trainer.strategy.barrier() # wait for all processes to reach this point

        # getting the best model from the previous trainer
        best_model_ckpt = trainer.checkpoint_callback.best_model_path
        
        with trainer.init_module(empty_init=True):
            model = LightningPLM.load_from_checkpoint(best_model_ckpt)

        trainer.test(model=model, dataloaders=val_dl)


        # if trainer.global_rank == 0:
        #     trainer = _get_trainer(config, devices=1)
        #     with trainer.init_module(empty_init=True):
        #         model = LightningPLM.load_from_checkpoint(best_model_ckpt)
        #     trainer.test(model=model, dataloaders=val_dl)

    else:
        assert config.load_from_checkpoint, "load_from_checkpoint is required for test_mode"
        trainer = _get_trainer(config, devices=1)
        with trainer.init_module(empty_init=True):
            model = LightningPLM.load_from_checkpoint(config.load_from_checkpoint)
        test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list)
        trainer.test(model=model, dataloaders=test_dl)

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
        trainer = _get_trainer(config)
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

def noise_removed_plm(config):
    if config.updated_train_dl_file:
        train_dl = torch.load(config.updated_train_dl_file, weights_only=False)
        log_info(logger, f"Loaded updated train_dl from {config.updated_train_dl_file}")
        log_info(logger, f"Total number of training samples: {len(train_dl.dataset)}")
        config.logging_dir = os.path.dirname(config.updated_train_dl_file)
    else:
        datamodule = DataModuleFromRaw(config)
        train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
        val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)
        
        
        hc_sample_ids, mc_sample_ids, lc_sample_ids = _find_noisy_samples_agentic(config, train_dl, val_dl)
        
        log_info(logger, f"HC: {len(hc_sample_ids)}, MC: {len(mc_sample_ids)}, LC: {len(lc_sample_ids)}")
        
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

        # save updated train_dl
        torch.save(train_dl, os.path.join(config.logging_dir, "updated_train_dl.pt"))

    
    # Train the model with the updated dataset
    vanilla_plm(config, train_dl=train_dl)
