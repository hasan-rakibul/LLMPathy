from omegaconf import OmegaConf
import lightning as L 
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import logging
from model import LightningPLM, LightningPLMFC
from utils import log_info

logger = logging.getLogger(__name__)

def init_model(config: OmegaConf) -> L.LightningModule:
    if config.use_demographics:
        return LightningPLMFC(config)
    else:
        return LightningPLM(config)
    
def load_model_from_ckpt(config: OmegaConf, ckpt: str) -> L.LightningModule:
    if config.use_demographics:
        return LightningPLMFC.load_from_checkpoint(ckpt)
    else:
        return LightningPLM.load_from_checkpoint(ckpt)
 

def get_trainer(config, devices="auto", extra_callbacks=None, enable_checkpointing=True, enable_early_stopping=True):
    """
    By default, we have EarlyStopping.
    ModelCheckpoint is enabled if enable_checkpointing is True.
    If you want to add more callbacks, pass them in extra_callbacks.
    """
    callbacks = []

    if enable_early_stopping:
        # early_stopping = EarlyStopping(
        #     monitor="val_pcc",
        #     patience=5,
        #     mode="max",
        #     min_delta=0.001
        # )
        # maybe val_pcc is not a good idea as pcc is not correlated beween val and test
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min",
            min_delta=0.001
        )
        callbacks.append(early_stopping)
    else:
        log_info(logger, "Early stopping disabled")

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
        max_epochs=1 if config.debug_mode else config.num_epochs,
        default_root_dir=config.logging_dir,
        deterministic=True,
        logger=True,
        log_every_n_steps=10,
        callbacks=callbacks,
        devices=devices,
        enable_checkpointing=enable_checkpointing,
        limit_train_batches=0.1 if config.debug_mode else 1.0,
    )

    return trainer