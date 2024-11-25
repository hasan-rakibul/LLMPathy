import os
import transformers
import lightning as L
from omegaconf import OmegaConf
from preprocess import DataModuleFromRaw
import logging
import torch
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef, mean_squared_error
import pandas as pd

from utils import resolve_logging_dir, log_info, read_file, resolve_seed_wise_checkpoint, process_seedwise_metrics, prepare_test_config
from model import load_model_from_ckpt

logger = logging.getLogger(__name__)

def test_plm(config: OmegaConf, have_label: bool = True) -> dict:
    assert os.path.exists(config.test_from_checkpoint), "valid test_from_checkpoint is required for test_mode"
    
    datamodule = DataModuleFromRaw(config)
    trainer = L.Trainer(
        logger=False,
        devices=1,
        max_epochs=1
    )

    with trainer.init_module(empty_init=True):
        model = load_model_from_ckpt(config, config.test_from_checkpoint)

    log_info(logger, f"Loaded model from {config.test_from_checkpoint}")
    
    test_dl = datamodule.get_test_dl(data_path_list=config.test_file_list, have_label=have_label)

    trainer.test(model=model, dataloaders=test_dl, verbose=True)

    if have_label:
        # metrics calculation is possibel only if we have labels
        metrics = {
            "test_pcc": trainer.callback_metrics["test_pcc"].item(),
            "test_ccc": trainer.callback_metrics["test_ccc"].item(),
            "test_rmse": trainer.callback_metrics["test_rmse"].item()
        }
    else:
        metrics = {}

    return metrics

def _test_multi_seeds(ckpt_parent_dir: str, config: OmegaConf, have_label: bool = True) -> None:
    results = []

    for seed in config.seeds:
        config.seed = seed
        log_info(logger, f"Current seed: {config.seed}")
        config.test_from_checkpoint = resolve_seed_wise_checkpoint(ckpt_parent_dir, seed)
        config.logging_dir = ckpt_parent_dir
        test_metrics = test_plm(config, have_label)
        
        if have_label:
            # then we have metrics
            test_metrics["seed"] = seed
            log_info(logger, f"Metrics: {test_metrics}")
            results.append(test_metrics)

    if have_label:
        save_as = os.path.join(ckpt_parent_dir, "results_test.csv")
        process_seedwise_metrics(results, save_as)

def _test_zero_shot(filepath: str, val_goldstandard_filepath: str = None) -> None:
    df = read_file(filepath)
    if val_goldstandard_filepath is not None:
        goldstandard = pd.read_csv(
            val_goldstandard_filepath, 
            sep='\t',
            header=None # had no header in the file
        )
        # first column is empathy
        goldstandard = goldstandard.rename(columns={0: "empathy"})
        df = pd.concat([df, goldstandard], axis=1)

    assert df["llm_empathy"].shape[0] == df["empathy"].shape[0], "The number of predictions and ground truth should be same"
    df = df[["empathy", "llm_empathy"]]
    log_info(logger, f"NaNs in the dataframe: {df.isna().sum().sum()}. Dropping them ...")
    df = df.dropna()
    y = torch.tensor(df["empathy"].to_numpy(), dtype=torch.float64)
    y_hat = torch.tensor(df["llm_empathy"].to_numpy(), dtype=torch.float64)
    pcc = pearson_corrcoef(y_hat, y).item()
    ccc = concordance_corrcoef(y_hat, y).item()
    rmse = mean_squared_error(y_hat, y, squared=False).item()
    pcc = round(pcc, 3)
    ccc = round(ccc, 3)
    rmse = round(rmse, 3)
    log_info(logger, f"PCC & CCC & RMSE: {pcc} & {ccc} & {rmse}")
    actual_filename = os.path.basename(filepath).split(".")[0]
    with open(f"logs/zero_shot/metrics_{actual_filename}.txt", "w") as f:
        f.write(f"PCC & CCC & RMSE: {pcc} & {ccc} & {rmse}")
    log_info(logger, f"Metrics saved to logs/metrics_zero_shot_{actual_filename}.txt")

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    
    config_test = OmegaConf.load("config/config_test.yaml")
    
    config_common = OmegaConf.load("config/config_common.yaml")
    config = OmegaConf.merge(config_common, config_test)

    config = prepare_test_config(config)
    
    if "test_from_checkpoint" in config:
        log_info(logger, f"Doing a single test using {config.test_from_checkpoint}")
        log_info(logger, f"Normal testing on {config.test_file_list}")
        config.logging_dir = resolve_logging_dir(config) # update customised logging_dir
        test_plm(config, config.have_label)
    elif "test_from_ckpts_parent_dir" in config:
        log_info(logger, f"Multi-seed testing from {config.test_from_ckpts_parent_dir}")
        _test_multi_seeds(config.test_from_ckpts_parent_dir, config, have_label=config.have_label)
    elif "test_zero_shot_file" in config:
        log_info(logger, f"Zero shot testing on {config.test_zero_shot_file}")
        if "val_goldstandard_file" in config:
            _test_zero_shot(config.test_zero_shot_file, config.val_goldstandard_file)
        else:
            _test_zero_shot(config.test_zero_shot_file)
    else:
        raise ValueError("Either test_file_list or test_zero_shot_file should be present in config")
