import os
import logging
import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils import log_info
from preprocess import DataModuleFromRaw
from model import LightningPLM

logger = logging.getLogger(__name__)

def vanilla_plm(config, train_dl=None):
    datamodule = DataModuleFromRaw(config)

    if train_dl is None:
        train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)

    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

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

    model = LightningPLM(config)

    if config.resume_from_checkpoint:
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=config.resume_from_checkpoint
        )
    else:
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl
        )

    trainer.test(model=model, dataloaders=val_dl)


# Multiple agentic approach
# def _get_sample_wise_errors(dataset, trainer):
#     outputs = trainer.predict(dataset)
#     # calculated squared error
#     squared_error = (outputs.predictions.squeeze() - outputs.label_ids)**2
#     return squared_error

# def _find_noisy_samples_agentic(config, train_dataset, datamodule):
#     models = [get_model(config).cuda() for _ in range(config.num_agents)]
#     all_errors = np.zeros((config.num_agents, len(train_dataset)))

#     raw_logging_dir = config.logging_dir # so that we don't nest
#     raw_checkpoint_dir = config.checkpoint_dir
#     for i, model in enumerate(models):
#         # customise logging_dir per agent
#         config.logging_dir = os.path.join(raw_logging_dir, f"agent_{i}")
#         if config.checkpoint_dir:
#             # customise checkpoint_dir per agent. Assume each agent has only one checkpoint
#             config.checkpoint_dir = os.path.join(raw_checkpoint_dir, f"agent_{i}")
#         trainer = _train_model(config, model, train_dataset, datamodule)
#         all_errors[i, :] = _get_sample_wise_errors(train_dataset, trainer)

#     threshold = np.percentile(all_errors, config.noise_level, axis=1) # agent-wise
#     hc_index, mc_index, lc_index = [], [], []

#     for i in range(all_errors.shape[1]): # iter over samples
#         errors = all_errors[:, i] # errors of all agents for a sample

#         # count the number of agents that agree on HC and LC
#         hc_count = np.sum(errors < threshold)

#         if hc_count == config.num_agents: # all agents agree for HC
#             hc_index.append(i)
#         elif hc_count == 0: # no agent agrees for HC --> all agree for LC
#             lc_index.append(i)
#         else:
#             mc_index.append(i) # Mixed agreements
    
#     config.logging_dir = raw_logging_dir
#     config.checkpoint_dir = raw_checkpoint_dir
#     if config.noise_level:
#         log_info(logger, f"HC: {len(hc_index)}, MC: {len(mc_index)}, LC: {len(lc_index)}")
#         log_info(logger, "Saving indices as npy files...")
#         np.save(os.path.join(config.logging_dir, "hc_index_" + str(config.noise_level) + ".npy"), hc_index)
#         np.save(os.path.join(config.logging_dir, "mc_index_" + str(config.noise_level) + ".npy"), mc_index)
#         np.save(os.path.join(config.logging_dir, "lc_index_" + str(config.noise_level) + ".npy"), lc_index)

#     return hc_index, mc_index, lc_index

# def _update_labels(sample, mc_set, lc_set):
#     idx = sample["sample_id"].item() # convert Tensor to scalar
#     if idx in mc_set or idx in lc_set:
#         sample["labels"] = sample["gpt_empathy"]
#     return sample

# def noise_removed_plm(config):
#     datamodule = DataModuleFromRaw(config)
#     train_dataset = _get_train_dataset(datamodule, config.train_file_list, send_label=True)

#     if config.mc_index_file and config.lc_index_file:
#         mc_index = list(np.load(config.mc_index_file))
#         lc_index = list(np.load(config.lc_index_file))
#         config.logging_dir = os.path.dirname(config.mc_index_file)
#     else:
#         _, mc_index, lc_index = _find_noisy_samples_agentic(config, train_dataset, datamodule)
#         config.checkpoint_dir = False # reset checkpoint_dir as it (if any) is for agents

#     # If MC and LC indices, use gpt_empathy, otherwise, use crowdsource_empathy, which we already have as labels
#     # Convert indices to sets for quick lookup
#     mc_set = set(mc_index)
#     lc_set = set(lc_index)
    
#     updated_train_dataset = train_dataset.map(
#         lambda sample: _update_labels(sample, mc_set, lc_set),
#         batched=False
#     )

#     # Save the updated dataset
#     updated_train_dataset.save_to_disk(os.path.join(config.logging_dir, "updated_train_dataset"))

#     vanilla_plm(config, train_dataset=updated_train_dataset)

    