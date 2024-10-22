from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.regression import PearsonCorrCoef, ConcordanceCorrCoef
from torchmetrics import MeanSquaredError
import os
import logging
import pandas as pd
import torch.nn as nn
from utils import log_info

logger = logging.getLogger(__name__)

# taken and modified from our LLM-GEm work
class RoBERTaFusionFC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        fc_arch = self.config.fc_arch

        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.config.plm,
            num_labels=fc_arch[0]
        )

        if config.freeze_plm:
            log_info(logger, "Freezing the PLM weights")
            for param in self.transformer.roberta.parameters():
                param.requires_grad = False
        
        # pre-fusion layer
        self.pre_fusion = nn.Sequential(
            nn.Tanh(), # as transformer outputs logits
            nn.Dropout(0.4),
            nn.Linear(fc_arch[0], fc_arch[1]),
            nn.Tanh(),
            nn.Dropout(0.3)
        )

        # fusion layer
        fusion_input_dim = fc_arch[-3] + len(config.demographics)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fc_arch[-2]),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        # output layer
        self.output = nn.Linear(fc_arch[-2], fc_arch[-1])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        demographic_vector=None
    ):

        x = self.transformer(
            input_ids= input_ids,
            attention_mask=attention_mask,
        )
        x = x.logits # shape: (batch_size, num_labels)
        
        x = self.pre_fusion(x)
        
        if demographic_vector is not None:
            x = torch.cat([x, demographic_vector], 1)        
        
        x = self.fusion(x)
        x = self.output(x)
        return x.squeeze(-1) # remove the last dimension
    
class LightningPLM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.learning_rate = self.config.lr # separate for lr tuning by lightning

        if "fc_arch" in self.config:
            raise NotImplementedError("FusionFC is forward is commented out")
            self.model = RoBERTaFusionFC(self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.config.plm,
                num_labels=1
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
    
    # def forward_roberta_fusion_fc(self, batch):
    #     if len(self.config.demographics) > 0:
    #         demographic_vector = torch.cat(
    #             [batch[col].unsqueeze(1) for col in self.config.demographics], dim=1
    #         ) # shape: (batch_size, num_demographics)
    #     else:
    #         demographic_vector = None
            
    #     return self.model(
    #         input_ids=batch['input_ids'],
    #         attention_mask=batch['attention_mask'],
    #         demographic_vector=demographic_vector
    #     )

    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        return output.logits.squeeze(-1) # remove the last dimension

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.config.adamw_beta1, self.config.adamw_beta2),
            eps=self.config.adamw_eps,
            weight_decay=self.config.adamw_weight_decay
        )

        if self.config.lr_scheduler_type == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode='min',
                patience=self.config.plateau_patience,
                factor=self.config.plateau_factor,
                threshold=self.config.plateau_threshold
            )
        elif self.config.lr_scheduler_type == "linear":
            num_training_step = self.config.num_training_steps
            lr_scheduler = get_linear_schedule_with_warmup(
                optimiser,
                num_warmup_steps=self.config.linear_warmup*num_training_step,
                num_training_steps=num_training_step
            )
        return {
            'optimizer': optimiser,
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
        outputs = self(batch)

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
        outputs = self(batch)
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
        outputs = self(batch)
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
