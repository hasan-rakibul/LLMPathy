import lightning as L
from omegaconf import OmegaConf
from preprocess import DataModuleFromRaw
from model import LightningPLM
from utils import resolve_logging_dir

def test(config):
    datamodule = DataModuleFromRaw(config)
    trainer = L.Trainer(
        logger=False,
        devices=1
    )
    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(config.load_from_checkpoint)
    test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list)
    trainer.test(model=model, dataloaders=test_dl)

if __name__ == "__main__":
    config = OmegaConf.load("config/config_test.yaml")
    assert config.load_from_checkpoint, "load_from_checkpoint is required for test_mode"

    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir

    test(config)