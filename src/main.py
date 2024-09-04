import os
import argparse
import datetime
from omegaconf import OmegaConf
from trainer import vanialla_plm, k_fold_cross_validation, find_noisy_samples_mcd, find_noisy_samples_agentic
from utils import seed_everything
from preprocess import DataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    seed_everything(42)

    if config.train.checkpoint_dir:
        assert os.path.exists(config.train.checkpoint_dir), "checkpoint_dir does not exist"
        logging_dir = config.train.checkpoint_dir   
    else:
        logging_dir=os.path.join(
            config.train.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
        os.makedirs(logging_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(logging_dir, "config.yaml"))

    config.train.logging_dir = logging_dir # update customised logging_dir

    datamodule = DataModule(config)
    train_dataset = datamodule.get_huggingface_data(
        data_file=config.data.train_file, 
        send_label=True
    )
    val_dataset = datamodule.get_huggingface_data(
        data_file=config.data.val_file,
        send_label=True
    )

    # vanialla_plm(config, train_dataset, val_dataset, datamodule)
    # k_fold_cross_validation(config, train_dataset, datamodule)
    # find_noisy_samples_mcd(config, train_dataset, datamodule)
    find_noisy_samples_agentic(config, train_dataset, datamodule)

if __name__ == "__main__":
    main()