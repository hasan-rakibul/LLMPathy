import os
import torch
import numpy as np
from transformers import RobertaModel
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

from preprocess import DataModule

def _get_embeddings(config):
    data_module = DataModule(config)
    dataloader = data_module.get_dataloader(data_file=config.data.train_file, send_label=True, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RobertaModel.from_pretrained(config.checkpoint)
    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu()) # shape: (batch_size, hidden_size [768])
            labels.append(batch['labels']) # shape: (batch_size, )
            
    embeddings = torch.cat(embeddings, dim=0) # shape: (num_samples, hidden_size)
    labels = torch.cat(labels, dim=0) # shape: (num_samples, )
    return embeddings, labels

def _compute_similarity(config):
    embeddings, labels = _get_embeddings(config)
    bin_edges = np.arange(0, 7.5, 0.5)
    # binned_labels = pd.cut(labels.numpy(), bins=bin_edges, labels=False, include_lowest=True)

    binned_labels = np.digitize(labels.numpy(), bin_edges) - 1 # map to 0-indexed bins

    all_similarities = []

    for bin_label in np.unique(binned_labels):
        # get indices of samples in the this bin
        bin_indices = np.where(binned_labels == bin_label)[0]

        if len(bin_indices) < 2:
            continue

        bin_embeddings = embeddings[bin_indices] # shape: (num_samples_in_bin, hidden_size)

         # Compute pairwise cosine similarity
        sim = cosine_similarity(bin_embeddings.unsqueeze(1), bin_embeddings.unsqueeze(0), dim=2)
        
        # Take only upper triangular part excluding the diagonal (since similarity with self is always 1)
        upper_tri_sim = sim.triu(diagonal=1)
        
        # Flatten the matrix to get all similarities
        similarities = upper_tri_sim[upper_tri_sim != 0].flatten().numpy()
        
        all_similarities.extend(similarities)

    return all_similarities


def plot_similarity(config):
    similarities = _compute_similarity(config)
    plt.hist(similarities, bins=1000)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('# of Essay Pairs')
    plt.savefig(os.path.join(config.train.logging_dir, 'Gaussian.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = OmegaConf.load('config/config.yaml')
    plot_similarity(config)
