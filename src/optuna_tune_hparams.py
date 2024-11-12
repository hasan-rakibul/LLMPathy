import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualization

import os
import datetime
import logging
import transformers
from omegaconf import OmegaConf
from functools import partial
import lightning as L

from utils import log_info, log_debug, get_trainer, read_file
from preprocess import DataModuleFromRaw
from model import LightningPLM

from agentic_noise_removal import _agentic_noise_removal

logger = logging.getLogger(__name__)


# following https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
def objective_agentic(trial: optuna.trial.Trial, config) -> float:

    # things to tune
    config.noise_level = trial.suggest_float("noise_level", 0.0, 1.0)
    config.num_agents = trial.suggest_int("num_agents", 2, 5)

    L.seed_everything(config.seed)

    model = LightningPLM(config)

    if len(config.objectives) > 1:
        # multi objective optimisation
        extra_callbacks = None
    else:
        extra_callbacks = [
            PyTorchLightningPruningCallback(trial, monitor="val_rmse")
        ]

    trainer = get_trainer(
        config,
        extra_callbacks=extra_callbacks,
        enable_checkpointing=False
    )

    train_dl = _agentic_noise_removal(config)

    datamodule = DataModuleFromRaw(config)
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    metrics = []
    for objective in config.objectives:
        metrics.append(trainer.callback_metrics[objective].item())
    
    return tuple(metrics)

def objective_llm_gem(trial: optuna.trial.Trial, config) -> float:

    # things to tune
    config.num_epochs = trial.suggest_int("num_epochs", 2, 50)
    config.lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    config.batch_size = trial.suggest_int("batch_size", 1, 32)
    config.max_length = trial.suggest_int("max_length", 128, 512, step=128)
    config.adamw_beta1 = trial.suggest_float("adamw_beta1", 0.8, 0.99)
    config.adamw_beta2 = trial.suggest_float("adamw_beta2", 0.8, 0.9999)
    config.adamw_eps = trial.suggest_float("adamw_eps", 1e-8, 1e-4, log=True)
    config.adamw_weight_decay = trial.suggest_float("adamw_weight_decay", 1e-5, 0.1, log=True)
    
    config.lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "plateau"])
    if config.lr_scheduler_type == "linear":
        config.linear_warmup = trial.suggest_float("linear_warmup", 0.0, 0.5)
    elif config.lr_scheduler_type == "plateau":
        config.plateau_patience = trial.suggest_int("plateau_patience", 1, 10)
        config.plateau_factor = trial.suggest_float("plateau_factor", 0.1, 0.9)
        config.plateau_threshold = trial.suggest_float("plateau_threshold", 1e-5, 1e-2)
    
    config.alpha = trial.suggest_float("alpha", 1.0, 6.0)

    L.seed_everything(config.seed)

    datamodule = DataModuleFromRaw(config)
    
    train_dl = datamodule.get_train_dl(data_path_list=config.train_file_list)
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    if config.lr_scheduler_type == "linear":
        # number of training steps is required for linear scheduler
        config.num_training_steps = len(train_dl) * config.num_epochs

    if len(config.objectives) > 1:
        # multi objective optimisation
        extra_callbacks = None
    else:
        extra_callbacks = [
            PyTorchLightningPruningCallback(trial, monitor="val_pcc")
        ]

    trainer = get_trainer(
        config,
        extra_callbacks=extra_callbacks,
        enable_checkpointing=True
    )

    with trainer.init_module():
        model = LightningPLM(config)

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    log_debug(logger, f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    log_debug(logger, f"Best model score: {trainer.callback_metrics["val_pcc"]}")

    best_model_ckpt = trainer.checkpoint_callback.best_model_path
    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(best_model_ckpt, config=config)
    trainer.validate(model=model, dataloaders=val_dl)

    log_debug(logger, f"Best model validation score: {trainer.callback_metrics['val_pcc']}")

    metrics = []
    for objective in config.objectives:
        metrics.append(trainer.callback_metrics[objective].item())
    
    return tuple(metrics)
    
if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    config_hparam = OmegaConf.load("config/config_hparam_tuner.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")
    config = OmegaConf.merge(config_common, config_hparam)

    if "resume_optuna_dir" in config:
        storage = f"sqlite:///{config.resume_optuna_dir}/optuna.db"
        log_info(logger, f"Resuming from {config.resume_optuna_dir}")
        config.logging_dir = config.resume_optuna_dir
    else:
        if "--debug_mode" in config:
            logger.setLevel(logging.DEBUG)
            config.logging_dir = "/tmp"
            log_info(logger, f"Running in debug mode. Logging to {config.logging_dir}")

        config.logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
        os.makedirs(config.logging_dir, exist_ok=True)
        storage = f"sqlite:///{config.logging_dir}/optuna.db"

    # atm, multi objective optimisation does not support pruning
    # but we have early stopping in the trainer, so it's fine
    pruner = optuna.pruners.NopPruner() if len(config.objectives) > 1 else optuna.pruners.MedianPruner()

    assert len(config.objectives) == len(config.directions), "Number of objectives and directions must match"

    study = optuna.create_study(
        study_name=config.expt_name,
        storage=storage,
        directions=config.directions,
        pruner=pruner,
        load_if_exists=True
    )

    study.set_metric_names(list(config.objectives)) # converted ListConfig to list, as it throws error

    # objective_param = partial(objective_agentic, config=config)
    objective_param = partial(objective_llm_gem, config=config)
    study.optimize(objective_param, n_trials=config.n_optuna_trails, show_progress_bar=False)

    trial_results = study.trials_dataframe()
    trial_results.to_csv(os.path.join(config.logging_dir, "trials_results.csv"))

    log_info(logger, f"Number of finished trials: {len(study.trials)}")
    if len(config.objectives) > 1:
        best_trials = study.best_trials
        with open(os.path.join(config.logging_dir, "best_trials_params.txt"), 'w') as f:
            for i, best_trail in enumerate(best_trials):
                f.write(f"Best trial {i}:\n")
                for key, value in best_trail.params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
    else:
        best_trial = study.best_trial
        log_info(logger, f"Best trial:{best_trial.value}")

        with open(os.path.join(config.logging_dir, "best_trial_params.txt"), 'w') as f:
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        
        # only for single objective optimisation
        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.write_image(os.path.join(config.logging_dir, "Optuna_slice.pdf"))

    fig_imp = optuna.visualization.plot_param_importances(study)
    fig_imp.write_image(os.path.join(config.logging_dir, "Optuna_param_importances.pdf"))
