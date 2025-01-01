import os
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

import sys

import torch.utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import DataModuleFromRaw
from utils import prepare_train_config

from tsne import LightningPLMWrapper

import scienceplots
plt.style.use(['science', 'tableau-colorblind10'])

def _compute_similarity_roberta(config, ckpt_path, threshold_diff):
    data_module = DataModuleFromRaw(config)
    
    df = data_module._raw_to_processed(
        path=config.file_for_similarity,
        have_label=True,
        mode="train"
    )

    dataloader = data_module.get_val_dl([config.file_for_similarity])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LightningPLMWrapper.load_from_checkpoint(ckpt_path, config=config)
    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu()) # shape: (batch_size, hidden_size [768])
            
    embeddings = torch.cat(embeddings, dim=0) # shape: (num_samples, hidden_size)

    labels = df[config.col_for_similarity].to_numpy()
    articles = df['article_id'].tolist()
    unique_articles = set(articles)

    all_similarities = []
    import pdb; pdb.set_trace()

    # Article-wise separation
    for article in unique_articles:
        # get indices for the same article
        indices = [i for i, a in enumerate(articles) if a == article]
        article_embeddings = [embeddings[i] for i in indices]
        article_labels = [labels[i] for i in indices]

        for i in range(len(article_labels)):
            for j in range(i+1, len(article_labels)):
                if abs(article_labels[i] - article_labels[j]) <= threshold_diff:
                    similarity = cosine_similarity(article_embeddings[i], article_embeddings[j])
                    all_similarities.append(similarity.item())
    
    # bin_edges = np.arange(0, 7.5, 0.5)

    # binned_labels = np.digitize(labels.numpy(), bin_edges) - 1 # map to 0-indexed bins

    # all_similarities = []

    # for bin_label in np.unique(binned_labels):
    #     # get indices of samples in the this bin
    #     bin_indices = np.where(binned_labels == bin_label)[0]

    #     if len(bin_indices) < 2:
    #         continue

    #     bin_embeddings = embeddings[bin_indices] # shape: (num_samples_in_bin, hidden_size)

    #      # Compute pairwise cosine similarity
    #     sim = cosine_similarity(bin_embeddings.unsqueeze(1), bin_embeddings.unsqueeze(0), dim=2)
        
    #     # Take only upper triangular part excluding the diagonal (since similarity with self is always 1)
    #     upper_tri_sim = sim.triu(diagonal=1)
        
    #     # Flatten the matrix to get all similarities
    #     similarities = upper_tri_sim[upper_tri_sim != 0].flatten().numpy()
        
    #     all_similarities.extend(similarities)

    return all_similarities

def subplot_similarities_roberta(config, bins, threshold_diff, ckpt_path_base, ckpt_path_mixed):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    config.col_for_similarity = "empathy"

    similarities = _compute_similarity_roberta(config, ckpt_path_base, threshold_diff)
    axs[0].hist(similarities, bins=bins)
    axs[0].set_xlabel('Cosine Similarity')
    axs[0].set_ylabel('Essay Pairs')
    axs[0].set_title('Crowdsourced')

    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 50])

    config.col_for_similarity = "llm_empathy"
    similarities = _compute_similarity_sbert(config, ckpt_path_mixed, threshold_diff)
    axs[1].hist(similarities, bins=bins)
    axs[1].set_xlabel('Cosine Similarity')
    axs[1].set_ylabel('Essay Pairs')
    axs[1].set_title('LLM')

    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 50])

    plt.savefig(os.path.join(config.logging_dir,\
            f'Gaussian_{config.train_data}_threshold_{threshold_diff}_article-wise.pdf'),
            format='pdf', bbox_inches='tight')

def _compute_similarity_sbert(config, threshold_diff):
    model = SentenceTransformer("all-mpnet-base-v2")
    data_module = DataModuleFromRaw(config)
    df = data_module._raw_to_processed(
        path=config.file_for_similarity,
        have_label=True,
        mode="train"
    )

    embeddings = model.encode(df['essay'].tolist(), show_progress_bar=True)
    labels = df[config.col_for_similarity].to_numpy()
    articles = df['article_id'].tolist()
    unique_articles = set(articles)

    all_similarities = []

    # Article-wise separation
    for article in unique_articles:
        # get indices for the same article
        indices = [i for i, a in enumerate(articles) if a == article]
        article_embeddings = [embeddings[i] for i in indices]
        article_labels = [labels[i] for i in indices]

        for i in range(len(article_labels)):
            for j in range(i+1, len(article_labels)):
                if abs(article_labels[i] - article_labels[j]) <= threshold_diff:
                    similarity = model.similarity(article_embeddings[i], article_embeddings[j])
                    all_similarities.append(similarity.item())

    # Without article-wise separation
    # for i in range(len(labels)):
    #     for j in range(i+1, len(labels)):
    #         if abs(labels[i] - labels[j]) <= threshold_diff:
    #             similarity = model.similarity(embeddings[i], embeddings[j])
    #             all_similarities.append(similarity.item())

    return all_similarities

def subplot_similarities_sbert(config, bins, threshold_diff):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    config.col_for_similarity = "empathy"

    similarities = _compute_similarity_sbert(config, threshold_diff)
    axs[0].hist(similarities, bins=bins)
    axs[0].set_xlabel('Cosine Similarity')
    axs[0].set_ylabel('Essay Pairs')
    axs[0].set_title('Crowdsourced')

    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 50])

    config.col_for_similarity = "llm_empathy"
    similarities = _compute_similarity_sbert(config, threshold_diff)
    axs[1].hist(similarities, bins=bins)
    axs[1].set_xlabel('Cosine Similarity')
    axs[1].set_ylabel('Essay Pairs')
    axs[1].set_title('LLM')

    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 50])

    plt.savefig(os.path.join(config.logging_dir,\
            f'Gaussian_{config.train_data}_threshold_{threshold_diff}_article-wise.pdf'),
            format='pdf', bbox_inches='tight')
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    config_train = OmegaConf.load("config/config_train.yaml")
    config_common = OmegaConf.load("config/config_common.yaml")

    config = OmegaConf.merge(config_common, config_train)
    raw_log_dir = config.logging_dir
    config = prepare_train_config(config)
    config.logging_dir = raw_log_dir # restore logging_dir as just logs/

    config.seed = config.seeds[0]
    config.batch_size = config.batch_sizes[0]

    config.extra_columns_to_keep = ["article_id", "llm_empathy"]

    config.file_for_similarity = "data/NewsEmp2024/trac3_EMP_train_gpt.tsv"
    
    subplot_similarities_sbert(config, bins=100, threshold_diff=0.5)


    # ckpt_path_base = "logs/20241115_004656_y(2024)-ImprovedEarlyStop/lr_3e-05_bs_16/seed_0/lightning_logs/version_18332572/checkpoints/epoch=9-step=630.ckpt"
    # ckpt_path_mixed = "logs/20241115_004406_y'(2024)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_0.0/seed_100/lightning_logs/version_18332544/checkpoints/epoch=7-step=504.ckpt"
    # subplot_similarities_roberta(config, bins=100, threshold_diff=0.5, ckpt_path_base=ckpt_path_base, ckpt_path_mixed=ckpt_path_mixed)