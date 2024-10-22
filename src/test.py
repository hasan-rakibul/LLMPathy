import os
import transformers
import lightning as L
from omegaconf import OmegaConf
from preprocess import DataModuleFromRaw
from model import LightningPLM
import zipfile
import logging
from utils import resolve_logging_dir, log_info

logger = logging.getLogger(__name__)

def _submission_ready(save_dir: str) -> None:
    # we need to zip the predictions
    with zipfile.ZipFile(f"{save_dir}/predictions.zip", "w") as zf:
        zf.write(f"{save_dir}/test-predictions_EMP.tsv", arcname="predictions_EMP.tsv")
        log_info(logger, f"Zipped predictions to {save_dir}/predictions.zip")

def _test(config):
    assert os.path.exists(config.load_from_checkpoint), "valid load_from_checkpoint is required for test_mode"
    
    datamodule = DataModuleFromRaw(config)
    trainer = L.Trainer(
        logger=False,
        devices=1,
        max_epochs=1
    )


    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(config.load_from_checkpoint)

    log_info(logger, f"Loaded model from {config.load_from_checkpoint}")
    
    if "2024" in config.test_file_list[0]:
        test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list, have_label=True)
    else:
        test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list)
    
    # modified setting from config do not work because we are loading a checkpoint with this value being False
    model.config.save_predictions_to_disk = config.save_predictions_to_disk
    model.config.logging_dir = config.logging_dir

    trainer.test(model=model, dataloaders=test_dl, verbose=True)

    if "--submission_ready" in config:
        _submission_ready(save_dir=config.logging_dir)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config_test = OmegaConf.load("config/config_test.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")
    config = OmegaConf.merge(config_common, config_test)
    
    L.seed_everything(config.seed)

    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir

    _test(config)
