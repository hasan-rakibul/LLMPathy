import os
import logging
import torch
import transformers
import lightning as L
from omegaconf import OmegaConf
import numpy as np

from utils import log_info, get_trainer, resolve_logging_dir, process_seedwise_metrics, prepare_config
from preprocess import DataModuleFromRaw
from model import LightningPLM
from test import test_plm

logger = logging.getLogger(__name__)

def _train_validate_plm(config, train_dl=None):
    datamodule = DataModuleFromRaw(config)
    
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

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
    
    if "resume_train_from_checkpoint" in config:
        log_info(logger, f"Resuming training from {config.resume_train_from_checkpoint}")
        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        with trainer.init_module():
            # model created here directly goes to GPU
            model = LightningPLM(config)

        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=config.resume_train_from_checkpoint
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
    
    # final validation at the end of training
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

def _run_multi_seeds(config: OmegaConf, do_test: bool = False) -> None:
    parent_logging_dir = config.logging_dir
    results = []
    for seed in config.seeds:
        config.seed = seed
        log_info(logger, f"Current seed: {config.seed}")
        config.logging_dir = os.path.join(parent_logging_dir, f"seed_{config.seed}")

        L.seed_everything(config.seed)

        best_model_ckpt, metrics = _train_validate_plm(config)
        
        if do_test or metrics["val_pcc"] > 0.5 or metrics["val_ccc"] > 0.5:
            # subsequent testing
            log_info(logger, f"Testing right after training from {best_model_ckpt}")
            config.test_from_checkpoint = best_model_ckpt
            config.logging_dir = resolve_logging_dir(config)
            test_metrics = test_plm(config)
            metrics = {**metrics, **test_metrics} # merge the two dictionaries 

        metrics["seed"] = seed
        log_info(logger, f"Metrics: {metrics}")
        results.append(metrics)
    save_as = os.path.join(parent_logging_dir, "results.csv")
    process_seedwise_metrics(results, save_as)

def _alpha_sweep(config: OmegaConf, do_test: bool) -> None:
    alpha_range = np.arange(0, 6.5, 0.5)
    parent_logging_dir = config.logging_dir

    # if you want to run for a specific range of alpha, like, to resume
    # alpha_range = np.arange(3.5, 6.5, 0.5)
    # parent_logging_dir = "logs/20241030_091522_LLMGEm+2023-and-2022"

    for alpha in alpha_range:
        config.logging_dir = os.path.join(parent_logging_dir, f"alpha_{alpha}")
        config.alpha = alpha.item() # converting to python float as numpy.float64 is not supported by OmegaConf
        log_info(logger, f"Current alpha: {config.alpha}")
        _run_multi_seeds(config, do_test)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config_train = OmegaConf.load("config/config_train.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")

    config = OmegaConf.merge(config_common, config_train)

    config = prepare_config(config)
    import pdb; pdb.set_trace()

    if "updated_train_dl_file" in config:
        train_dl = torch.load(config.updated_train_dl_file, weights_only=False)
        log_info(logger, f"Loaded updated train_dl from {config.updated_train_dl_file}")
        log_info(logger, f"Total number of training samples: {len(train_dl.dataset)}")
        config.logging_dir = os.path.dirname(config.updated_train_dl_file)
        _ = _train_validate_plm(config, train_dl)

    if "--debug_mode" in config:
        logger.setLevel(logging.DEBUG)
        config.seeds = config.seeds[:2] # reduce the number of seeds for debugging
        config.logging_dir = "./tmp"
        log_info(logger, f"Debug mode is on. Using {config.logging_dir} for storing log files.")
    
    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir
    
    if config.main_label == "y":
        if not config.tune_hparams:
            _run_multi_seeds(config, do_test=config.do_test)
        else:
            parent_logging_dir = config.logging_dir
            for lr in config.lrs:
                config.lr = lr
                for batch_size in config.batch_sizes:
                    config.batch_size = batch_size
                    log_info(logger, f"Current lr: {config.lr}, Current batch_size: {config.batch_size}")
                    config.logging_dir = os.path.join(parent_logging_dir, f"lr_{config.lr}_bs_{config.batch_size}")
                    _run_multi_seeds(config, do_test=config.do_test)
    elif config.main_label == "y'":
        assert "llm_empathy" in config.extra_columns_to_keep_train, "llm_empathy must be included in extra_columns_to_keep_train"
        if not config.tune_hparams:
            _alpha_sweep(config, do_test=config.do_test)
        else:
            parent_logging_dir = config.logging_dir
            for lr in config.lrs:
                config.lr = lr
                for batch_size in config.batch_sizes:
                    config.batch_size = batch_size
                    log_info(logger, f"Current lr: {config.lr}, Current batch_size: {config.batch_size}")
                    config.logging_dir = os.path.join(parent_logging_dir, f"lr_{config.lr}_bs_{config.batch_size}")
                    _alpha_sweep(config, do_test=config.do_test)
    else:
        raise ValueError(f"main_label must be either y or y'. Found {config.main_label}")
