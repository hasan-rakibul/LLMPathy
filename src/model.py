from transformers import AutoModelForSequenceClassification
import torch
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.regression import PearsonCorrCoef, ConcordanceCorrCoef
from torchmetrics import MeanSquaredError
import os
import logging
import pandas as pd
from utils import log_info

logger = logging.getLogger(__name__)

class LightningPLM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.config.plm,
            num_labels=1 # regression
        )
        self.training_step_outputs = []
        self.training_step_labels = []
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []
        self.pcc = PearsonCorrCoef()
        self.ccc = ConcordanceCorrCoef()
        self.rmse = MeanSquaredError(squared=False) # root mean squared error

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits.squeeze(-1) # remove the last dimension
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.98),
            eps=1e-06,
            weight_decay=0.1
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            threshold=0.001
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    
    def _outputs_from_batch(self, batch):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        return outputs
    
    @rank_zero_only
    def _calc_save_predictions(self, preds, labels=None, mode='test'):
        preds_np = preds.cpu().numpy()
        
        pred_df = pd.DataFrame({'emp': preds_np, 'dis': preds_np}) # we're not predicting distress, just aligning with submission system
        pred_df.to_csv(
            os.path.join(self.config.logging_dir, f"{mode}-predictions_EMP.tsv"),
            sep='\t', index=None, header=None
        )
        log_info(logger, f'Saved predictions to {self.config.logging_dir}/{mode}-predictions_EMP.tsv')
        
        if labels is not None:
            pcc_score = self.pcc(preds, labels).item()
            ccc_score = self.ccc(preds, labels).item()
            rmse_score = self.rmse(preds, labels).item()
            pcc_score = round(pcc_score, 3)
            ccc_score = round(ccc_score, 3)
            rmse_score = round(rmse_score, 3)

            with open(os.path.join(self.config.logging_dir, f"{mode}-metrics.txt"), 'w') as f:
                f.write(f"PCC: {pcc_score}\n")
                f.write(f"CCC: {ccc_score}\n")
                f.write(f"RMSE: {rmse_score}\n")

    def training_step(self, batch, batch_idx):
        outputs = self._outputs_from_batch(batch)
        loss = torch.nn.functional.mse_loss(outputs, batch['labels'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.training_step_outputs.append(outputs)
        self.training_step_labels.append(batch['labels'])

        return loss
    
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.training_step_outputs)
        all_labels = torch.cat(self.training_step_labels)
        self.log(
            'train_pcc', 
            self.pcc(all_preds, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            'train_ccc',
            self.ccc(all_preds, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            'train_rmse',
            self.rmse(all_preds, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        self.training_step_outputs.clear() # free up memory
        self.training_step_labels.clear()
    
    def validation_step(self, batch, batch_idx):
        outputs = self._outputs_from_batch(batch)
        loss = torch.nn.functional.mse_loss(outputs, batch['labels'])
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.append(outputs)
        self.validation_step_labels.append(batch['labels'])
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        all_labels = torch.cat(self.validation_step_labels)
        self.log(
            'val_pcc',
            self.pcc(all_preds, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            'val_ccc',
            self.ccc(all_preds, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            'val_rmse',
            self.rmse(all_preds, all_labels),
            logger=True,
            prog_bar=True,
            sync_dist=True
        )

        if self.config.save_predictions_to_disk:
            self._calc_save_predictions(all_preds, all_labels, mode='val')

        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    # cannot be made to run on a single GPU
    def test_step(self, batch, batch_idx):
        outputs = self._outputs_from_batch(batch)
        self.test_step_outputs.append(outputs)
        if 'labels' in batch:
            self.test_step_labels.append(batch['labels'])
    
    @rank_zero_only
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_outputs)
        if len(self.test_step_labels) > 0:
            all_labels = torch.cat(self.test_step_labels)
        else:
            all_labels = None

        if self.config.save_predictions_to_disk:
            self._calc_save_predictions(all_preds, all_labels, mode='test')

        self.test_step_outputs.clear()
        self.test_step_labels.clear()
    
    