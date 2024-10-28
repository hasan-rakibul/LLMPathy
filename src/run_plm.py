import os
import logging
import torch
import transformers
import lightning as L
from omegaconf import OmegaConf
import pandas as pd

from utils import log_info, get_trainer, resolve_logging_dir
from preprocess import DataModuleFromRaw
from model import LightningPLM
from test import test_plm

logger = logging.getLogger(__name__)

def train_vanilla_plm(config, train_dl=None):
    datamodule = DataModuleFromRaw(config)
    
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    if not config.val_only:    
        trainer = get_trainer(config)
        if train_dl is None:
            train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)

        if config.lr_scheduler_type == "linear":
            # number of training steps is required for linear scheduler
            config.num_training_steps = len(train_dl) * config.num_epochs

        if config.lr_find: # didn't work well in a casual run
            raise NotImplementedError("lr_find needs to be re-configured as the model initialisation is moved to later part")
            # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
            tuner = L.pytorch.tuner.Tuner(trainer)
            lr_finder = tuner.lr_find(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
            fig = lr_finder.plot(suggest=True)
            fig.savefig(os.path.join(config.logging_dir, "lr_finder.png"))
            log_info(logger, f"lr_finder plot saved at {config.logging_dir}/lr_finder.png")
            config.lr = lr_finder.suggestion() # update config
            model.learning_rate = lr_finder.suggestion() # update model
        
        if "load_from_checkpoint" in config:
            log_info(logger, f"Resuming training from {config.load_from_checkpoint}")
            # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
            with trainer.init_module():
                # model created here directly goes to GPU
                model = LightningPLM(config)

            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                ckpt_path=config.load_from_checkpoint
            )
        elif "finetune_from_checkpoint" in config:
            log_info(logger, f"Fine-tuning from {config.finetune_from_checkpoint}")
            assert "alpha" not in config, "alpha should not be provided for fine-tuning"
            assert "train_file_only_LLM_list" not in config, "train_file_only_LLM_list should not be provided for fine-tuning"
            with trainer.init_module(empty_init=True):
                model = LightningPLM.load_from_checkpoint(config.finetune_from_checkpoint)
            
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
            )
        else:
            log_info(logger, "Training from scratch")
            # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
            with trainer.init_module():
                # model created here directly goes to GPU
                model = LightningPLM(config)
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
        
    log_info(logger, f"Loading the best model from {best_model_ckpt}")
    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(best_model_ckpt)

    # model.config.save_predictions_to_disk = True # save final predictions to disk
    trainer.validate(model=model, dataloaders=val_dl)

    metrics = {
        "val_pcc": trainer.callback_metrics["val_pcc"].item(),
        "val_ccc": trainer.callback_metrics["val_ccc"].item(),
        "val_rmse": trainer.callback_metrics["val_rmse"].item()
    }

    return best_model_ckpt, metrics

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config_train = OmegaConf.load("config/config_train.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")

    config = OmegaConf.merge(config_common, config_train)

    if "updated_train_dl_file" in config:
        train_dl = torch.load(config.updated_train_dl_file, weights_only=False)
        log_info(logger, f"Loaded updated train_dl from {config.updated_train_dl_file}")
        log_info(logger, f"Total number of training samples: {len(train_dl.dataset)}")
        config.logging_dir = os.path.dirname(config.updated_train_dl_file)
        _ = train_vanilla_plm(config, train_dl)

    if "--debug_mode" in config:
        logger.setLevel(logging.DEBUG)
        # delete the logs
        config.logging_dir = "./tmp"
        log_info(logger, f"Debug mode is on. Using {config.logging_dir} for storing log files.")

    parent_logging_dir = resolve_logging_dir(config) # update customised logging_dir
    results = []
    for seed in config.seeds:
        config.seed = seed
        log_info(logger, f"Current seed: {config.seed}")
        config.logging_dir = os.path.join(parent_logging_dir, f"seed_{config.seed}")

        L.seed_everything(config.seed)
        best_model_ckpt, metrics = train_vanilla_plm(config)

        if "--do_test" in config:
            config_test = OmegaConf.load("config/config_test.yaml")
            config_test = OmegaConf.merge(config_common, config_test)
            config_test.load_from_checkpoint = best_model_ckpt
            config_test.logging_dir = resolve_logging_dir(config_test)
            config_test.seed = seed
            test_metrics = test_plm(config_test)
            metrics = {**metrics, **test_metrics} # merge the two dictionaries

        metrics["seed"] = seed
        log_info(logger, f"Metrics: {metrics}")
        results.append(metrics)
    
    results_df = pd.DataFrame(results)

    results_df.set_index("seed", inplace=True)
    results_df = results_df.round(3)
    
    # post-processing
    mean_row = results_df.mean(numeric_only=True).round(3)
    std_row = results_df.std(numeric_only=True).round(3)
    median_row = results_df.median(numeric_only=True).round(3)
    
    # Assign a label to identify each row
    mean_row.name = "mean"
    std_row.name = "std"
    median_row.name = "median"

    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T, median_row.to_frame().T])
    
    results_df.to_csv(os.path.join(parent_logging_dir, "results.csv"), index=True)
    log_info(logger, f"Results saved at {parent_logging_dir}/results.csv")

    # print the result, in LaTeX-table style
    log_info(logger, "Val PCC & Val CCC & Val RMSE & Test PCC & Test CCC & Test RMSE (mean +/- std)")
    log_info(logger, " & ".join([f"${mean:.3f}\\pm {std:.3f}$"\
            for mean, std in zip(mean_row, std_row)]))
