import os
import logging
import torch
import transformers
import lightning as L
from omegaconf import OmegaConf

from utils import log_info, get_trainer, resolve_logging_dir
from preprocess import DataModuleFromRaw
from model import LightningPLM

logger = logging.getLogger(__name__)

def tain_vanilla_plm(config, train_dl=None):
    datamodule = DataModuleFromRaw(config)
    
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    if not config.val_only:    
        trainer = get_trainer(config)
        if train_dl is None:
            train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
    
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

        # getting the best model from the previous trainer
        best_model_ckpt = trainer.checkpoint_callback.best_model_path
    else:
        assert config.load_from_checkpoint, "load_from_checkpoint must be provided for validation only mode"
        best_model_ckpt = config.load_from_checkpoint
        # very basic trainer for validation only mode
        trainer = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(best_model_ckpt)

    model.config.save_predictions_to_disk = True # save final predictions to disk
    trainer.validate(model=model, dataloaders=val_dl)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config = OmegaConf.load("config/config_train.yaml")
    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir

    L.seed_everything(config.seed)

    if config.updated_train_dl_file:
        train_dl = torch.load(config.updated_train_dl_file, weights_only=False)
        log_info(logger, f"Loaded updated train_dl from {config.updated_train_dl_file}")
        log_info(logger, f"Total number of training samples: {len(train_dl.dataset)}")
        config.logging_dir = os.path.dirname(config.updated_train_dl_file)
        tain_vanilla_plm(config, train_dl)
    else:
        tain_vanilla_plm(config)
        