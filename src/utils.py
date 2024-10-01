import os
import datetime
import logging
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


import matplotlib.pyplot as plt
import scienceplots

from lightning.pytorch.utilities import rank_zero_only

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def get_trainer(config, devices="auto", callbacks=None):
    if callbacks is None:
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min"
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                min_delta=0.01
            )
        ]

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        default_root_dir=config.logging_dir,
        deterministic=True,
        logger=True,
        log_every_n_steps=10,
        callbacks=callbacks,
        devices=devices
    )

    return trainer

def resolve_logging_dir(config):
    if config.load_from_checkpoint:
        assert os.path.exists(config.load_from_checkpoint), "checkpoint_dir does not exist"
        logging_dir = config.load_from_checkpoint   
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
