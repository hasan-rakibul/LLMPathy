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

from utils import log_info, get_trainer
from preprocess import DataModuleFromRaw
from model import LightningPLM

from agentic_noise_removal import _agentic_noise_removal

logger = logging.getLogger(__name__)


# following https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
def objective(trial: optuna.trial.Trial, config) -> float:

    # things to tune
    config.noise_level = trial.suggest_float("noise_level", 0.0, 1.0)
    config.num_agents = trial.suggest_int("num_agents", 1, 5)

    L.seed_everything(config.seed)

    model = LightningPLM(config)

    if len(config.objectives) > 1:
        # multi objective optimisation
        extra_callbacks = None
    else:
        extra_callbacks = [
            PyTorchLightningPruningCallback(trial, monitor="val_loss")
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

    
if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    config = OmegaConf.load("config/config_hparam_tuner.yaml")

    if config.resume_optuna_dir:
        storage = f"sqlite:///{config.resume_optuna_dir}/optuna.db"
        config.logging_dir = config.resume_optuna_dir
    else:
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

    objective_param = partial(objective, config=config)
    study.optimize(objective_param, n_trials=config.n_optuna_trails, show_progress_bar=True)

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

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(config.logging_dir, "Optuna_param_importances.pdf"))
