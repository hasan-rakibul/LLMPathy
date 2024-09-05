import os
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import torch
from utils import resolve_checkpoint
from preprocess import DataModule
from model import get_model

def _get_train_dataset(datamodule, file_path, send_label=True):
    dataset = datamodule.get_huggingface_data(
        data_file=file_path, 
        send_label=send_label
    )
    return dataset

def _compute_metrics(eval_pred):
    pearsonr_metric = evaluate.load('pearsonr')
    logits, labels = eval_pred # unpack the tuple returned by the model
    pearsonr = pearsonr_metric.compute(predictions=logits, references=labels)
    return pearsonr

def _train_model(config, model, train_dataset, datamodule):
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
        # gradient_checkpointing_kwargs={'use_reentrant':True} # recompute the forward pass if True
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

def _train_validate_model(config, model, train_dataset, val_dataset, datamodule):
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
        # gradient_checkpointing_kwargs={'use_reentrant':True} # recompute the forward pass if True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=datamodule.data_collator,
        compute_metrics=_compute_metrics
    )

    if config.train.checkpoint_dir:
        checkpoint = resolve_checkpoint(config.train.checkpoint_dir)
        print(f"\nResuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    return trainer

def _save_predictions(config, dataset, eval):
    y_pred = eval.predictions.squeeze().tolist()
    y = dataset["labels"].tolist()
    pearsonr = eval.metrics['test_pearsonr']
    print(f"Validation Pearsonr: {pearsonr:.4f}")

    # if dataset has no labels
    if "labels" in dataset.column_names:
        df = pd.DataFrame({
            "sample_index": dataset["__index_level_0__"].tolist(),
            "predictions": y_pred,
            "ground_truth": y
        })
    else:
        df = pd.DataFrame({
            "sample_index": dataset["__index_level_0__"].tolist(),
            "predictions": y_pred
        })

    df.to_csv(os.path.join(config.train.logging_dir, "predictions.csv"), index=False)

    # save pearson r to a file
    with open(os.path.join(config.train.logging_dir, "pearsonr.txt"), "w") as f:
        f.write(str(pearsonr))

def vanilla_plm(config, train_dataset=None):
    model = get_model(config)
    # model = model.cuda()

    datamodule = DataModule(config)
    if train_dataset is None:
        train_dataset = _get_train_dataset(datamodule, config.data.train_file, send_label=True)
    
    val_dataset = _get_train_dataset(datamodule, config.data.val_file, send_label=True)

    trainer = _train_validate_model(config, model, train_dataset, val_dataset, datamodule)

    eval_results = trainer.predict(val_dataset)
    _save_predictions(config, val_dataset, eval_results)

def k_fold_cross_validation(config):
    from sklearn.model_selection import KFold

    model = get_model(config)
    # model = model.cuda()

    datamodule = DataModule(config)
    train_dataset = _get_train_dataset(datamodule, config.data.train_file, send_label=True)
    
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
        trainer = _train_validate_model(config, model, train_dataset, val_dataset, datamodule)
        
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
def _enable_mc_dropout(model):
    """Enable dropout layers during inference."""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def _mc_dropout_predict(model, dataloader, num_samples=50):
    """Perform Monte Carlo Dropout predictions."""
    model.eval()
    _enable_mc_dropout(model)

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

def find_noisy_samples_mcd(config):

    model = get_model(config)
    # model = model.cuda()

    datamodule = DataModule(config)
    train_dataset = _get_train_dataset(datamodule, config.data.train_file, send_label=True)

    trainer = _train_model(config, model, train_dataset, datamodule)

    # Use DataLoader for the training set
    train_dataloader = trainer.get_eval_datasetloader(train_dataset)

    # Perform MC Dropout predictions on the training set
    mean_preds, var_preds = _mc_dropout_predict(model, train_dataloader, num_samples=10)

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
def _get_sample_wise_errors(dataset, trainer):
    outputs = trainer.predict(dataset)
    # calculated squared error
    squared_error = (outputs.predictions.squeeze() - outputs.label_ids)**2
    return squared_error

def _find_noisy_samples_agentic(config, train_dataset, datamodule):
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
        trainer = _train_model(config, model, train_dataset, datamodule)
        all_errors[i, :] = _get_sample_wise_errors(train_dataset, trainer)

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
    
    config.train.logging_dir = raw_logging_dir
    config.train.checkpoint_dir = raw_checkpoint_dir
    if config.data.noise_level:
        print(f"HC: {len(hc_index)}, MC: {len(mc_index)}, LC: {len(lc_index)}")
        print("Saving indices as npy files...")
        np.save(os.path.join(config.train.logging_dir, "hc_index_" + str(config.data.noise_level) + ".npy"), hc_index)
        np.save(os.path.join(config.train.logging_dir, "mc_index_" + str(config.data.noise_level) + ".npy"), mc_index)
        np.save(os.path.join(config.train.logging_dir, "lc_index_" + str(config.data.noise_level) + ".npy"), lc_index)

    return hc_index, mc_index, lc_index

def _update_labels(sample, mc_set, lc_set):
    idx = sample["__index_level_0__"].item() # convert Tensor to scalar
    if idx in mc_set or idx in lc_set:
        sample["labels"] = sample["gpt_empathy"]
    return sample

def noise_removed_plm(config):
    datamodule = DataModule(config)
    train_dataset = _get_train_dataset(datamodule, config.data.train_file, send_label=True)

    if config.data.mc_index_file and config.data.lc_index_file:
        mc_index = list(np.load(config.data.mc_index_file))
        lc_index = list(np.load(config.data.lc_index_file))
    else:
        _, mc_index, lc_index = _find_noisy_samples_agentic(config, train_dataset, datamodule)
        config.train.checkpoint_dir = False # reset checkpoint_dir as it (if any) is for agents

    # If MC and LC indices, use gpt_empathy, otherwise, use crowdsource_empathy, which we already have as labels
    # Convert indices to sets for quick lookup
    mc_set = set(mc_index)
    lc_set = set(lc_index)
    
    updated_train_dataset = train_dataset.map(
        lambda sample: _update_labels(sample, mc_set, lc_set),
        batched=False
    )

    # Save the updated dataset
    updated_train_dataset.save_to_disk(os.path.join(config.train.logging_dir, "updated_train_dataset"))

    vanilla_plm(config, train_dataset=updated_train_dataset)

    