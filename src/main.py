import os
import argparse
import datetime
from omegaconf import OmegaConf
from utils import seed_everything
from preprocess import DataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

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

    # from trainer import vanilla_plm
    # vanilla_plm(config)

    # from trainer import k_fold_cross_validation
    # k_fold_cross_validation(config)

    # from trainer import find_noisy_samples_mcd
    # find_noisy_samples_mcd(config)

    from trainer import noise_removed_plm
    noise_removed_plm(config)

if __name__ == "__main__":
    main()