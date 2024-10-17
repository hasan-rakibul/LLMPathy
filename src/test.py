import transformers
import lightning as L
from omegaconf import OmegaConf
from preprocess import DataModuleFromRaw
from model import LightningPLM
import zipfile
import logging
from utils import resolve_logging_dir, log_info

logger = logging.getLogger(__name__)

def _submission_ready(config):
    # we need to zip the predictions
    with zipfile.ZipFile(f"{config.logging_dir}/predictions.zip", "w") as zf:
        zf.write(f"{config.logging_dir}/test-predictions_EMP.tsv", arcname="predictions_EMP.tsv")
        log_info(logger, f"Zipped predictions to {config.logging_dir}/predictions.zip")

def _test(config):
    datamodule = DataModuleFromRaw(config)
    trainer = L.Trainer(
        logger=False,
        devices=1,
        max_epochs=1
    )
    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(config.load_from_checkpoint)
    test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list)
    
    # modified setting from config do not work because we are loading a checkpoint with this value being False
    model.config.save_predictions_to_disk = True
    model.config.logging_dir = config.logging_dir

    trainer.test(model=model, dataloaders=test_dl, verbose=True)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config_test = OmegaConf.load("config/config_test.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")
    config = OmegaConf.merge(config_common, config_test)
    
    L.seed_everything(config.seed)
    assert config.load_from_checkpoint, "load_from_checkpoint is required for test_mode"

    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir

    _test(config)

    if config.submission_ready:
        _submission_ready(config)