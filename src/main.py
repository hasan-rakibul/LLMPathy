import os
import argparse
import datetime
from omegaconf import OmegaConf
import lightning as L
import logging
from trainer import vanilla_plm

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    L.seed_everything(42)

    if config.resume_from_checkpoint:
        assert os.path.exists(config.resume_from_checkpoint), "checkpoint_dir does not exist"
        logging_dir = config.resume_from_checkpoint   
    else:
        logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )

    config.logging_dir = logging_dir # update customised logging_dir

    vanilla_plm(config)


if __name__ == "__main__":
    main()