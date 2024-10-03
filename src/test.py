import transformers
import lightning as L
from omegaconf import OmegaConf
from preprocess import DataModuleFromRaw
from model import LightningPLM
from utils import resolve_logging_dir

def test(config):
    datamodule = DataModuleFromRaw(config)
    trainer = L.Trainer(
        logger=False,
        devices=1,
        max_epochs=1
    )
    with trainer.init_module(empty_init=True):
        model = LightningPLM.load_from_checkpoint(config.load_from_checkpoint)
    test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list)
    
    # setting this from config does not work because we are loading a checkpoint with this value being False
    model.config.save_predictions_to_disk = True

    trainer.test(model=model, dataloaders=test_dl, verbose=True)

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config = OmegaConf.load("config/config_test.yaml")
    L.seed_everything(config.seed)
    assert config.load_from_checkpoint, "load_from_checkpoint is required for test_mode"

    config.logging_dir = resolve_logging_dir(config) # update customised logging_dir

    test(config)