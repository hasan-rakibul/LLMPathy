import os
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import torch
from utils import resolve_checkpoint

from model import get_model

def compute_metrics(eval_pred):
    pearsonr_metric = evaluate.load('pearsonr')
    logits, labels = eval_pred # unpack the tuple returned by the model
    pearsonr = pearsonr_metric.compute(predictions=logits, references=labels)
    return pearsonr

def train_model(config, model, train_dataset, datamodule):
    """Just train, no validation"""
    training_args = TrainingArguments(
        output_dir=config.train.logging_dir,
        learning_rate=config.train.lr,
        lr_scheduler_type=config.train.lr_scheduler_type,
        warmup_ratio=config.train.warmup_ratio,
        max_grad_norm=config.train.max_grad_norm,
        num_train_epochs=config.train.num_epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        weight_decay=config.train.weight_decay,
        save_strategy='epoch',
        save_total_limit=1,
        logging_dir=config.train.logging_dir,
        logging_strategy='epoch',
        seed=config.seed,
        fp16=config.train.fp16,
        gradient_checkpointing=config.train.gradient_checkpointing,
        report_to='tensorboard',
        gradient_checkpointing_kwargs={'use_reentrant':True} # recompute the forward pass if True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=datamodule.data_collator,
    )

    if config.train.checkpoint_dir:
        checkpoint = resolve_checkpoint(config.train.checkpoint_dir)
        print(f"\nResuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    return trainer

def train_validate_model(config, model, train_dataset, val_dataset, datamodule):
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    training_args = TrainingArguments(
        output_dir=config.train.logging_dir,
        learning_rate=config.train.lr,
        lr_scheduler_type=config.train.lr_scheduler_type,
        warmup_ratio=config.train.warmup_ratio,
        max_grad_norm=config.train.max_grad_norm,
        num_train_epochs=config.train.num_epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        weight_decay=config.train.weight_decay,
        save_strategy='epoch',
        save_total_limit=1,
        logging_dir=config.train.logging_dir,
        logging_strategy='epoch',
        eval_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='pearsonr',
        greater_is_better=True,
        seed=config.seed,
        fp16=config.train.fp16,
        gradient_checkpointing=config.train.gradient_checkpointing,
        report_to='tensorboard',
        gradient_checkpointing_kwargs={'use_reentrant':True} # recompute the forward pass if True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_datasetset=val_dataset,
        data_collator=datamodule.data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer

def vanialla_plm(config, train_dataset, val_dataset, datamodule):
    model = get_model(config)
    model = model.cuda()
    _ = train_validate_model(config, train_dataset, val_dataset, datamodule)

def k_fold_cross_validation(config, train_dataset, datamodule):
    from sklearn.model_selection import KFold

    model = get_model(config)
    model = model.cuda()

    kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)

    predictions = []
    sample_index = []
    ground_truth = []

    raw_logging_dir = config.train.logging_dir
    for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        train_dataset = train_dataset.select(train_index)
        val_dataset = train_dataset.select(val_index)

        # customise logging_dir per fold
        config.train.logging_dir = os.path.join(raw_logging_dir, f"fold_{fold}")
        trainer = train_validate_model(config, model, train_dataset, val_dataset, datamodule)
        
        # Evaluate the trained model on the validation dataset
        eval_results = trainer.predict(val_dataset)

        # Get the predictions, samples, and ground truth
        predictions.append(eval_results.predictions.squeeze().tolist())
        sample_index.append(val_dataset["__index_level_0__"].tolist())
        ground_truth.append(val_dataset["labels"].tolist())

    # Save the predictions, samples, and ground truth
    df = pd.DataFrame({
        "sample_index": [item for sublist in sample_index for item in sublist],
        "predictions": [item for sublist in predictions for item in sublist],
        "ground_truth": [item for sublist in ground_truth for item in sublist]
    })

    df.to_csv(os.path.join(config.train.logging_dir, "k_fold_cross_validation_results.csv"), index=False)


# Monte Carlo Dropout
def enable_mc_dropout(model):
    """Enable dropout layers during inference."""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def mc_dropout_predict(model, dataloader, num_samples=50):
    """Perform Monte Carlo Dropout predictions."""
    model.eval()
    enable_mc_dropout(model)

    all_predictions = []

    for _ in range(num_samples):
        predictions = []
        for batch in dataloader:
            with torch.no_grad():
                outputs = model(**{k: v.cuda() for k, v in batch.items()})
            predictions.append(outputs.logits.cpu().numpy())
        all_predictions.append(np.concatenate(predictions))
    all_predictions = np.stack(all_predictions)  # Shape: (num_samples, num_batches, batch_size)
    mean_predictions = np.mean(all_predictions, axis=0)  # Mean over the samples
    variance_predictions = np.var(all_predictions, axis=0)  # Variance over the samples

    return mean_predictions, variance_predictions

def find_noisy_samples_mcd(config, train_dataset, datamodule):

    model = get_model(config)
    model = model.cuda()

    trainer = train_model(config, model, train_dataset, datamodule)

    # Use DataLoader for the training set
    train_dataloader = trainer.get_eval_datasetloader(train_dataset)

    # Perform MC Dropout predictions on the training set
    mean_preds, var_preds = mc_dropout_predict(model, train_dataloader, num_samples=10)

    # Flatten predictions and variance
    mean_preds = mean_preds.squeeze().tolist()
    var_preds = var_preds.squeeze().tolist()

    # Save the results
    df = pd.DataFrame({
        "sample_index": train_dataset["__index_level_0__"].tolist(),
        "predictions": mean_preds,
        "uncertainties": var_preds,
        "ground_truth": train_dataset["labels"].tolist()
    })

    df.to_csv(os.path.join(config.train.logging_dir, "mc_dropout_noisy_samples.csv"), index=False)

# Multiple agentic approach
def get_sample_wise_errors(dataset, trainer):
    outputs = trainer.predict(dataset)
    # calculated squared error
    squared_error = (outputs.predictions.squeeze() - outputs.label_ids)**2
    return squared_error

def find_noisy_samples_agentic(config, train_dataset, datamodule):
    models = [get_model(config).cuda() for _ in range(config.train.num_agents)]
    all_errors = np.zeros((config.train.num_agents, len(train_dataset)))

    raw_logging_dir = config.train.logging_dir # so that we don't nest
    raw_checkpoint_dir = config.train.checkpoint_dir
    for i, model in enumerate(models):
        # customise logging_dir per agent
        config.train.logging_dir = os.path.join(raw_logging_dir, f"agent_{i}")
        if config.train.checkpoint_dir:
            # customise checkpoint_dir per agent. Assume each agent has only one checkpoint
            config.train.checkpoint_dir = os.path.join(raw_checkpoint_dir, f"agent_{i}")
        trainer = train_model(config, model, train_dataset, datamodule)
        all_errors[i, :] = get_sample_wise_errors(train_dataset, trainer)

    threshold = np.percentile(all_errors, config.data.noise_level, axis=1) # agent-wise
    hc_index, mc_index, lc_index = [], [], []

    for i in range(all_errors.shape[1]): # iter over samples
        errors = all_errors[:, i] # errors of all agents for a sample

        # count the number of agents that agree on HC and LC
        hc_count = np.sum(errors < threshold)

        if hc_count == config.train.num_agents: # all agents agree for HC
            hc_index.append(i)
        elif hc_count == 0: # no agent agrees for HC --> all agree for LC
            lc_index.append(i)
        else:
            mc_index.append(i) # Mixed agreements
    
    # save indices
    np.save(os.path.join(raw_logging_dir, "hc_index_" + str(config.data.noise_level) + ".npy"), hc_index)
    np.save(os.path.join(raw_logging_dir, "mc_index_" + str(config.data.noise_level) + ".npy"), mc_index)
    np.save(os.path.join(raw_logging_dir, "lc_index_" + str(config.data.noise_level) + ".npy"), lc_index)
    