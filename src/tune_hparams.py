import optuna
from optuna.integration import PyTorchLightningPruningCallback

import os
import logging
from omegaconf import OmegaConf
from functools import partial

from utils import log_info, get_trainer, resolve_logging_dir
from preprocess import DataModuleFromRaw
from model import LightningPLM

from agentic_noise_removal import _agentic_noise_removal

logger = logging.getLogger(__name__)


# following https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
def objective(trial: optuna.trial.Trial, config) -> float:

    # things to tune
    config.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config.noise_level = trial.suggest_float("noise_level", 0.1, 0.9)
    config.num_agents = trial.suggest_int("num_agents", 1, 5)

    model = LightningPLM(config)
    trainer = get_trainer(
        config,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_ccc"),
        ]
    )

    train_dl = _agentic_noise_removal(config)

    datamodule = DataModuleFromRaw(config)
    val_dl = datamodule.get_val_dl(data_path_list=config.val_file_list)

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    return trainer.callback_metrics["val_ccc"].item()

    
if __name__ == "__main__":
    config = OmegaConf.load("config/config.yaml")
    config.logging_dir = resolve_logging_dir(config)

    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )

    objective_param = partial(objective, config=config)
    study.optimize(objective_param, n_trials=config.n_optuna_trails, show_progress_bar=True)

    log_info(logger, f"Number of finished trials: {len(study.trials)}")
    trial = study.best_trial
    log_info(logger, f"Best trial:{trial.value}")

    with open(os.path.join(config.logging_dir, "best_trial_params.txt"), 'w') as f:
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
            