import os
import evaluate
from transformers import TrainingArguments, Trainer

from preprocess import DataModule
from model import get_model

def compute_metrics(eval_pred):
    pearsonr_metric = evaluate.load('pearsonr')
    logits, labels = eval_pred # unpack the tuple returned by the model
    pearsonr = pearsonr_metric.compute(predictions=logits, references=labels)
    return pearsonr

def train_model(config):
    datamodule = DataModule(config)
    train_data = datamodule.get_huggingface_data(
        data_file=config.data.train_file, 
        send_label=True
    )
    val_data = datamodule.get_huggingface_data(
        data_file=config.data.val_file,
        send_label=True
    )

    model = get_model(config)
    model = model.cuda()

    output_dir = os.path.join(config.train.output_dir, config.checkpoint, config.expt_name)
    logging_dir = os.path.join(config.train.logging_dir, config.checkpoint, config.expt_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.train.lr,
        lr_scheduler_type=config.train.lr_scheduler_type,
        warmup_ratio=config.train.warmup_ratio,
        max_grad_norm=config.train.max_grad_norm,
        num_train_epochs=config.train.num_train_epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        weight_decay=config.train.weight_decay,
        save_strategy='epoch',
        save_total_limit=1,
        logging_dir=logging_dir,
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='pearsonr',
        greater_is_better=True,
        seed=config.seed,
        fp16=config.train.fp16,
        gradient_checkpointing=config.train.gradient_checkpointing,
        report_to='tensorboard',
        gradient_checkpointing_kwargs={'use_reentrant':False} # based on https://github.com/huggingface/transformers/issues/26969
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=datamodule.data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()