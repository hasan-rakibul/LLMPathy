import os
import logging
import transformers
import lightning as L
from omegaconf import OmegaConf
import warnings
import glob
import torch

from utils import log_info, resolve_logging_dir, process_seedwise_metrics, prepare_train_config, get_trainer, resolve_num_steps
from model import init_model, load_model_from_ckpt
from preprocess import DataModuleFromRaw
from test import test_plm

logger = logging.getLogger(__name__)

def _train_validate_plm(config: OmegaConf, train_dl: torch.utils.data.DataLoader = None) -> tuple:
    datamodule = DataModuleFromRaw(config)
    # if os.path.exists(config.logging_dir):
    #     log_info(logger, f"Seed-level logging directory already exists: {config.logging_dir}. So, validating on the saved ckpt...")
    #     # don't need the train dl --> need for tine-tuning 
    if train_dl is None:
        train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
    else:
        log_info(logger, "Training data loader is provided. So, skipping the training data loader creation.")
    
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    trainer = get_trainer(config, enable_early_stopping=config.enable_early_stopping)

    if config.lr_scheduler_type == "linear" or config.lr_scheduler_type == "polynomial":
        config.num_training_steps, config.num_warmup_steps = resolve_num_steps(config, train_dl)

    if config.lr_find: # didn't work well in a casual run
        model = init_model(config)
        warnings.warn("lr_find is not tested well. So, be cautious.")
        # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
        tuner = L.pytorch.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(config.logging_dir, "lr_finder.png"))
        log_info(logger, f"lr_finder plot saved at {config.logging_dir}/lr_finder.png")
        config.lr = lr_finder.suggestion() # update config
    
    if "resume_train_from_checkpoint" in config:
        # lowest level of resume
        log_info(logger, f"Resuming training from {config.resume_train_from_checkpoint}")
        warnings.warn("Optimiser / scheduler states are not managed I think. So, be cautious.")
        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        with trainer.init_module():
            # model created here directly goes to GPU
            model = init_model(config)

        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=config.resume_train_from_checkpoint
        )
    elif "finetune_from_checkpoint" in config:
        config.lr = config.finetune_lr
        log_info(logger, f"Fine-tuning from {config.finetune_from_checkpoint}")
        with trainer.init_module(empty_init=True):
            model = load_model_from_ckpt(config, config.finetune_from_checkpoint)

        log_info(logger, f"Fine-tuning LR: {model.learning_rate}")

        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl
        )
    else:
        log_info(logger, "Training from scratch")
        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html

        if not os.path.exists(config.logging_dir):
            with trainer.init_module():
                # model created here directly goes to GPU
                model = init_model(config) 
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
            )
    
    if os.path.exists(config.logging_dir):
        ckpt_list = glob.glob(os.path.join(config.logging_dir, "**/*.ckpt"), recursive=True)
        assert len(ckpt_list) == 1, f"Number of ckpt is not 1."
        best_model_ckpt = ckpt_list[0]
    else:
        # getting the best model from the previous trainer
        best_model_ckpt = trainer.checkpoint_callback.best_model_path
    
    # final validation at the end of training
    log_info(logger, f"Loading the best model from {best_model_ckpt}")
    with trainer.init_module(empty_init=True):
        model = load_model_from_ckpt(config, best_model_ckpt)

    # model.config.save_predictions_to_disk = True # save final predictions to disk
    trainer.validate(model=model, dataloaders=val_dl)

    metrics = {
        "val_pcc": trainer.callback_metrics["val_pcc"].item(),
        "val_ccc": trainer.callback_metrics["val_ccc"].item(),
        "val_rmse": trainer.callback_metrics["val_rmse"].item()
    }

    return best_model_ckpt, metrics

def _seeds_sweep(config: OmegaConf, do_test: bool = False, train_dl: torch.utils.data.DataLoader = None) -> None:
    parent_logging_dir = config.logging_dir
    results = []
    for seed in config.seeds:
        config.seed = seed
        log_info(logger, f"Current seed: {config.seed}")
        config.logging_dir = os.path.join(parent_logging_dir, f"seed_{config.seed}")

        L.seed_everything(config.seed)

        best_model_ckpt, metrics = _train_validate_plm(config, train_dl=train_dl)
        
        if do_test:
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
    parent_logging_dir = config.logging_dir

    for alpha in config.alphas:
        config.logging_dir = os.path.join(parent_logging_dir, f"alpha_{alpha}")
        config.alpha = alpha
        log_info(logger, f"Current alpha: {config.alpha}")
        if os.path.exists(config.logging_dir):
            log_info(logger, f"Alpha-level logging directory already exists: {config.logging_dir}")
            if os.path.exists(os.path.join(config.logging_dir, "results.csv")):
                log_info(logger, f"Alpha-level results.csv exists. So, skipping for this alpha.")
                continue
            else:
                log_info(logger, f"Results do not exist for this alpha. So, resuming for the remaining seeds.")
        _seeds_sweep(config, do_test)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    # config_train = OmegaConf.load("config/config_train.yaml")
    config_train = OmegaConf.load("config/Giorgi2024Findings.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")

    config = OmegaConf.merge(config_common, config_train)

    config = prepare_train_config(config)

    if config.debug_mode:
        logger.setLevel(logging.DEBUG)
        config.seeds = config.seeds[:2] # reduce the number of seeds for debugging
        config.logging_dir = "./tmp"
        log_info(logger, f"Debug mode is on. Using {config.logging_dir} for storing log files.")
        if config.main_label == "y_agentic":
            config.num_agents = 2
            log_info(logger, f"Debug mode is on. Using {config.num_agents} agents for agentic noise removal.")

    if "overwrite_logging_dir" in config:
        log_info(logger, f"Using overwrite_logging_dir {config.overwrite_logging_dir}")
        log_info(logger, "MAKE SURE you DELETE the last directory manually which was not trained for all epochs.")
        config.logging_dir = config.overwrite_logging_dir
    else:
        config.logging_dir = resolve_logging_dir(config) # update customised logging_dir
    
    if config.main_label == "y":
        parent_logging_dir = config.logging_dir
        for lr in config.lrs:
            config.lr = lr
            for batch_size in config.batch_sizes:
                config.batch_size = batch_size
                log_info(logger, f"Current lr: {config.lr}, Current batch_size: {config.batch_size}")
                config.logging_dir = os.path.join(parent_logging_dir, f"lr_{config.lr}_bs_{config.batch_size}")
                _seeds_sweep(config, do_test=config.do_test)
    elif config.main_label == "y'":
        parent_logging_dir = config.logging_dir
        for lr in config.lrs:
            config.lr = lr
            for batch_size in config.batch_sizes:
                config.batch_size = batch_size
                log_info(logger, f"Current lr: {config.lr}, Current batch_size: {config.batch_size}")
                config.logging_dir = os.path.join(parent_logging_dir, f"lr_{config.lr}_bs_{config.batch_size}")
                _alpha_sweep(config, do_test=config.do_test)
    
    elif config.main_label == "y_agentic":
        from agentic_noise_removal import agentic_noise_removal
        if config.updated_train_dl_file:
            assert os.path.exists(config.updated_train_dl_file), f"Updated train_dl file not found at {config.updated_train_dl_file}"
            train_dl = torch.load(config.updated_train_dl_file, weights_only=False)
            log_info(logger, f"Loaded updated train_dl from {config.updated_train_dl_file}")
            config.logging_dir = os.path.dirname(config.updated_train_dl_file)
        else:
            log_info(logger, "No updated train_dl file found. So, training from scratch.")
            config.batch_size = config.batch_sizes[0] # only the first batch_size is used for agentic
            config.lr = config.lrs[0] # only the first lr is used for agentic
            train_dl = agentic_noise_removal(config)

        parent_logging_dir = config.logging_dir
        for lr in config.lrs:
            config.lr = lr
            for batch_size in config.batch_sizes:
                config.batch_size = batch_size
                log_info(logger, f"Current lr: {config.lr}, Current batch_size: {config.batch_size}")
                config.logging_dir = os.path.join(parent_logging_dir, f"lr_{config.lr}_bs_{config.batch_size}")
                _seeds_sweep(config, do_test=config.do_test, train_dl=train_dl)
    else:
        raise ValueError(f"main_label must be either y or y'. Found {config.main_label}")
