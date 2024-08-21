import argparse
import omegaconf

from trainer import train_model
from utils import set_all_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config)

    set_all_seeds(42)

    train_model(config)

if __name__ == "__main__":
    main()