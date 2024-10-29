import os
from typing import List
import datetime
import logging
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import scienceplots

import pandas as pd
import glob

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
        # 2024 raw data are in csv. The essay has commas, and placed inside double quotes
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
    # early_stopping = EarlyStopping(
    #     monitor="val_pcc",
    #     patience=5,
    #     mode="max",
    #     min_delta=0.001
    # )
    # maybe val_pcc is not a good idea as pcc is not correlated beween val and test

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        min_delta=0.001
    )
    callbacks = []
    
    callbacks.append(early_stopping)
    # log_info(logger, "Early stopping disabled")
    
    if enable_checkpointing:
        # have a ModelCheckpoint callback
        # callbacks.append(
        #     ModelCheckpoint(
        #         monitor="val_pcc",
        #         save_top_k=1,
        #         mode="max"
        #     )
        # )
        # checkpoint = ModelCheckpoint(
        #     monitor="val_loss",
        #     save_top_k=1,
        #     mode="min"
        # )
        checkpoint = ModelCheckpoint(
            save_top_k=1 # saves the last checkpoint; no need to save_last=True as it will save another checkpoint unnecessarily
        )
        
        callbacks.append(checkpoint)
        
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
    if "resume_train_from_checkpoint" in config:
        assert os.path.exists(config.resume_train_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.resume_train_from_checkpoint.split("/")[:-4] # logs/.../ ; calculate from end as we may have ./logs/ or just logs/
        logging_dir = os.path.join(*path_list)   
    elif "finetune_from_checkpoint" in config:
        assert os.path.exists(config.finetune_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.finetune_from_checkpoint.split("/")[:-4]
        # add additional directory for finetuning
        path_list.append("finetune")
        logging_dir = os.path.join(*path_list)
    elif "test_from_checkpoint" in config:
        assert os.path.exists(config.test_from_checkpoint), "checkpoint_dir does not exist"
        path_list = config.test_from_checkpoint.split("/")[:-4]
        logging_dir = os.path.join(*path_list)
    else:
        logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
    return logging_dir

def resolve_seed_wise_checkpoint(parent_dir: str, seed: float) -> str:
    ckpt = glob.glob(os.path.join(parent_dir, f"seed_{seed}/**/*.ckpt"), recursive=True)
    assert len(ckpt) == 1, f"Found {len(ckpt)} checkpoints for seed {seed}"
    return ckpt[0]

def process_seedwise_metrics(results: List, save_as: str) -> None:
    results_df = pd.DataFrame(results)

    results_df.set_index("seed", inplace=True)
    results_df = results_df.round(3)
    
    # post-processing
    mean_row = results_df.mean(numeric_only=True).round(3)
    std_row = results_df.std(numeric_only=True).round(3)
    median_row = results_df.median(numeric_only=True).round(3)
    
    # Assign a label to identify each row
    mean_row.name = "mean"
    std_row.name = "std"
    median_row.name = "median"

    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T, median_row.to_frame().T])
    
    results_df.to_csv(save_as, index=True)
    log_info(logger, f"Results saved at {save_as}")

    # print the result, in LaTeX-table style
    log_info(logger, "[Val, Test] PCC & CCC & RMSE (mean +/- std)")
    log_info(logger, " & ".join([f"${mean:.3f}\\pm {std:.3f}$"\
            for mean, std in zip(mean_row, std_row)]))

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
