from transformers import AutoModelForSequenceClassification
import torch
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.regression import PearsonCorrCoef, ConcordanceCorrCoef
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
        self.pearsonr = PearsonCorrCoef()
        self.ccc = ConcordanceCorrCoef()

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
    
    def _save_predictions(self, preds, labels=None):
        preds_np = preds.cpu().numpy()
        
        pred_df = pd.DataFrame({'emp': preds_np, 'dis': preds_np}) # we're not predicting distress, just aligning with submission system
        pred_df.to_csv(
            os.path.join(self.config.logging_dir, "predictions_EMP.tsv"),
            sep='\t', index=None, header=None
        )
        log_info(logger, f'Saved predictions to {self.config.logging_dir}/predictions_EMP.tsv')
        
        if labels is not None:
            pearsonr = self.pearsonr(preds, labels)
            ccc = self.ccc(preds, labels)
            pearsonr = pearsonr.cpu().numpy()
            ccc = ccc.cpu().numpy()
            log_info(logger, f'Validation pearsonr: {pearsonr}')
            log_info(logger, f'Validation CCC: {ccc}')

            with open(os.path.join(self.config.logging_dir, "metrics.txt"), 'w') as f:
                f.write(f"Pearsonr: {pearsonr}\n")
                f.write(f"CCC: {ccc}\n")

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
            'train_pearsonr', 
            self.pearsonr(all_preds, all_labels),
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
            'val_pearsonr',
            self.pearsonr(all_preds, all_labels),
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

        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    @rank_zero_only
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

        self._save_predictions(all_preds, all_labels)

        self.test_step_outputs.clear()
        self.test_step_labels.clear()
        