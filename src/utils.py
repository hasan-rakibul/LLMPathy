import os
import datetime
import logging
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import scienceplots

import pandas as pd

from lightning.pytorch.utilities import rank_zero_only

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep='\t', na_values="unknown") # some column includes "unknown"
    elif file_path.endswith(".csv"):
        # 2024 data is csv. The essay has commas, and placed inside double quotes
        # Further, tt has \" inside the quoted text, for example, "I am a \"good\" person"
        df = pd.read_csv(file_path, quotechar='"', escapechar="\\")

        # "2024" has different column names
        df = df.rename(columns={
            "person_essay": "essay",
            "person_empathy": "empathy"
        })
    else:
        raise ValueError(f"File extension not supported: {file_path}")
    return df

def get_trainer(config, devices="auto", extra_callbacks=None, enable_checkpointing=True):
    """
    By default, we have EarlyStopping.
    ModelCheckpoint is enabled if enable_checkpointing is True.
    If you want to add more callbacks, pass them in extra_callbacks.
    """
    # early stopping callback is always there
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
            min_delta=0.01
        )
    ]
    callbacks = [] # remove early stopping for now
    log_info(logger, "Early stopping disabled")
    
    if enable_checkpointing:
        # have a ModelCheckpoint callback
        callbacks.append(
            ModelCheckpoint(
                monitor="val_pcc",
                save_top_k=1,
                mode="max"
            )
        )
    
    callbacks.extend(extra_callbacks) if extra_callbacks else None

    trainer = L.Trainer(
        max_epochs=1 if "--debug_mode" in config else config.num_epochs,
        default_root_dir=config.logging_dir,
        deterministic=True,
        logger=True,
        log_every_n_steps=10,
        callbacks=callbacks,
        devices=devices,
        enable_checkpointing=enable_checkpointing,
        limit_train_batches=0.1 if "--debug_mode" in config else 1.0,
    )

    return trainer

def resolve_logging_dir(config):
    if "load_from_checkpoint" in config:
        assert os.path.exists(config.load_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.load_from_checkpoint.split("/")[:-4] # logs/.../ ; calculate from end as we may have ./logs/ or just logs/
        logging_dir = os.path.join(*path_list)   
    else:
        logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
    return logging_dir

@rank_zero_only
def log_info(logger, msg):
    logger.info(msg)

@rank_zero_only
def log_debug(logger, msg):
    logger.debug(msg)

def plot(x, y, y2=None, xlabel=None, ylabel=None, legend=[], save=False, filename=None):
    """Plot data points"""
    plt.style.use(['science'])
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(x, y)
    if y2 is not None:
        ax.plot(x, y2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    
    if save:
        plt.savefig(fname=filename+'.pdf', format='pdf', bbox_inches='tight')
        print(f"Saved as {filename}.pdf")
        
    fig.show()
