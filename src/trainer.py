import os
import logging
import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils import log_info
from preprocess import DataModuleFromRaw
from model import LightningPLM

logger = logging.getLogger(__name__)

def _get_trainer(config):
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
        ]
    )

    return trainer

def vanilla_plm(config, train_dl=None):
    
    trainer = _get_trainer(config)

    datamodule = DataModuleFromRaw(config)

    if not config.test_mode:
        if train_dl is None:
            train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)

        val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

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

        trainer.test(model=model, dataloaders=val_dl)

    else:
        assert config.load_from_checkpoint, "load_from_checkpoint is required for test_mode"
        model = LightningPLM.load_from_checkpoint(config.load_from_checkpoint)
        test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list)
        trainer.test(model=model, dataloaders=test_dl)


def _find_noisy_samples_agentic(config, train_dl, val_dl):
    models = [LightningPLM(config) for _ in range(config.num_agents)]
    all_errors = np.zeros((config.num_agents, len(train_dl.dataset)))

    raw_logging_dir = config.logging_dir # so that we don't nest
    for i, model in enumerate(models):
        # customise logging_dir per agent
        config.logging_dir = os.path.join(raw_logging_dir, f"agent_{i}")
        trainer = _get_trainer(config)
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        # automatically auto-loads the best weights from the previous run
        preds = trainer.predict(train_dl)
        # calculated squared error
        all_errors[i, :] = (preds - train_dl.dataset["labels"]) ** 2

    threshold = np.percentile(all_errors, config.noise_level, axis=1) # agent-wise
    hc_index, mc_index, lc_index = [], [], []

    for i in range(all_errors.shape[1]): # iter over samples
        errors = all_errors[:, i] # errors of all agents for a sample

        # count the number of agents that agree on HC and LC
        hc_count = np.sum(errors < threshold)

        if hc_count == config.num_agents: # all agents agree for HC
            hc_index.append(i)
        elif hc_count == 0: # no agent agrees for HC --> all agree for LC
            lc_index.append(i)
        else:
            mc_index.append(i) # Mixed agreements
    
    config.logging_dir = raw_logging_dir
    if config.noise_level:
        log_info(logger, f"HC: {len(hc_index)}, MC: {len(mc_index)}, LC: {len(lc_index)}")
        log_info(logger, "Saving indices as npy files...")
        np.save(os.path.join(config.logging_dir, "hc_index_" + str(config.noise_level) + ".npy"), hc_index)
        np.save(os.path.join(config.logging_dir, "mc_index_" + str(config.noise_level) + ".npy"), mc_index)
        np.save(os.path.join(config.logging_dir, "lc_index_" + str(config.noise_level) + ".npy"), lc_index)

    return hc_index, mc_index, lc_index

def noise_removed_plm(config):
    datamodule = DataModuleFromRaw(config)
    train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    if config.mc_index_file and config.lc_index_file:
        mc_index = list(np.load(config.mc_index_file))
        lc_index = list(np.load(config.lc_index_file))
        config.logging_dir = os.path.dirname(config.mc_index_file)
    else:
        assert config.load_from_checkpoint == False, "Checkpointing is not implemented for agents"
        _, mc_index, lc_index = _find_noisy_samples_agentic(config, train_dl, val_dl)

    # If MC and LC indices, use llm_empathy, otherwise, use crowdsource_empathy, which we already have as labels
    # Convert indices to sets for quick lookup
    mc_set = set(mc_index)
    lc_set = set(lc_index)
    
    # If sample_id is in mc_set or lc_set, update the label to llm_empathy
    for batch in train_dl:
        for i, sample_id in enumerate(batch["sample_id"]):
            sample_id = sample_id.item() # index tensor to scalar
            if sample_id in mc_set or sample_id in lc_set:
                batch["labels"][i] = batch[config.llm_column][i]

    log_info(logger, f"Updated labels for {len(mc_set) + len(lc_set)} samples")
    
    # Train the model with the updated dataset
    vanilla_plm(config, train_dl=train_dl)
