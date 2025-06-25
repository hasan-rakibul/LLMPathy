import os
import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
import numpy as np

from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# the final plot was done in local OpenSUSE
# since Setonix didn't have latex installed
# Comment the following to run non-latex version in Setonix
import scienceplots
plt.style.use(['science', 'tableau-colorblind10'])

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import LightningPLM
from preprocess import DataModuleFromRaw

class LightningPLMWrapper(LightningPLM):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, batch):
        output = self.model.roberta(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        return output

def _get_embeddings(config, filepath, ckpt_path, n_tsne = 3):
    model = LightningPLMWrapper.load_from_checkpoint(ckpt_path, config=config)

    dm = DataModuleFromRaw(config)
    dl = dm.get_val_dl(filepath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            output = model(batch)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu())
            labels.append(batch["labels"])

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    tsne = TSNE(
        n_components=n_tsne,
        perplexity=30,
        random_state=config.seed,
        method="exact",
        n_jobs=-1,
        verbose=1
    )
    tsne_embeddings = tsne.fit_transform(embeddings.numpy())

    return embeddings.cpu().numpy(), tsne_embeddings, labels.cpu().numpy()

def _get_embeddings_additional(config, ckpt_path, n_tsne = 3):
    model = LightningPLMWrapper.load_from_checkpoint(ckpt_path, config=config)

    base_data = config[2024].train
    config.train_file_only_LLM_list = [config[2022].train_llama, config[2022].val_llama]
    config.train_only_llm_portion = 1.0

    dm = DataModuleFromRaw(config)
    dl = dm.get_train_dl([base_data])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            output = model(batch)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu())
            labels.append(batch["labels"])

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    tsne = TSNE(
        n_components=n_tsne,
        perplexity=30,
        random_state=config.seed,
        method="exact",
        n_jobs=-1,
        verbose=1
    )
    tsne_embeddings = tsne.fit_transform(embeddings.numpy())

    return embeddings.cpu().numpy(), tsne_embeddings, labels.cpu().numpy()

def _plot_tsne3d(config, file_base: list, file_mixed: list, ckpt_path_base: str, ckpt_path_mixed: str, ckpt_path_add: str):
    if os.path.exists("logs/tsne/embeddings_base_model.npy") and os.path.exists("logs/tsne/labels_base.npy") and os.path.exists("logs/tsne/embeddings_base_tsne.npy"):
        embeddings_base_model = np.load("logs/tsne/embeddings_base_model.npy")
        embeddings_base = np.load("logs/tsne/embeddings_base_tsne.npy")
        labels_base = np.load("logs/tsne/labels_base.npy")
    else:
        embeddings_base_model, embeddings_base, labels_base = _get_embeddings(config, file_base, ckpt_path_base)
        np.save("logs/tsne/embeddings_base_model.npy", embeddings_base_model)
        np.save("logs/tsne/embeddings_base_tsne.npy", embeddings_base)
        np.save("logs/tsne/labels_base.npy", labels_base)
    
    config.label_column = config.llm_column
    if os.path.exists("logs/tsne/embeddings_mixed_model.npy") and os.path.exists("logs/tsne/embeddings_mixed_tsne.npy") and os.path.exists("logs/tsne/labels_mixed.npy"):
        embeddings_mixed_model = np.load("logs/tsne/embeddings_mixed_model.npy")
        embeddings_mixed = np.load("logs/tsne/embeddings_mixed_tsne.npy")
        labels_mixed = np.load("logs/tsne/labels_mixed.npy")
    else:
        embeddings_mixed_model, embeddings_mixed, labels_mixed = _get_embeddings(config, file_mixed, ckpt_path_mixed)
        np.save("logs/tsne/embeddings_mixed_model.npy", embeddings_mixed_model)
        np.save("logs/tsne/embeddings_mixed_tsne.npy", embeddings_mixed)
        np.save("logs/tsne/labels_mixed.npy", labels_mixed)

    config.label_column = "empathy" # reset
    if os.path.exists("logs/tsne/embeddings_add_model.npy") and os.path.exists("logs/tsne/embeddings_add_tsne.npy") and os.path.exists("logs/tsne/labels_add.npy"):
        embeddings_add_model = np.load("logs/tsne/embeddings_add_model.npy")
        embeddings_add = np.load("logs/tsne/embeddings_add_tsne.npy")
        labels_add = np.load("logs/tsne/labels_add.npy")
    else:
        embeddings_add_model, embeddings_add, labels_add = _get_embeddings_additional(config, ckpt_path_add)
        np.save("logs/tsne/embeddings_add_model.npy", embeddings_add_model)
        np.save("logs/tsne/embeddings_add_tsne.npy", embeddings_add)
        np.save("logs/tsne/labels_add.npy", labels_add)

    fig = plt.figure(figsize=(14, 6), constrained_layout=False)

    norm = plt.Normalize(1, 7)
    cmap = plt.colormaps["managua"]

    ax1 = fig.add_subplot(131, projection="3d")

    score_base = _tsne3d_metrics(embeddings_base_model, labels_base)
    print(f"Silhouette score (base): {score_base}")

    sc_base = ax1.scatter(
        embeddings_base[:, 0], embeddings_base[:, 1], embeddings_base[:, 2], c=labels_base, cmap=cmap, norm=norm
    )
    ax1.set_title("NewsEmp24 Crowdsourced\n" + r"\textbf{Silhouette score: $\mathbf{" + f"{score_base}" + r"}$}")

    ax2 = fig.add_subplot(132, projection="3d")
    score_mixed = _tsne3d_metrics(embeddings_mixed_model, labels_mixed)
    print(f"Silhouette score (mixed): {score_mixed}")

    _ = ax2.scatter(
        embeddings_mixed[:, 0], embeddings_mixed[:, 1], embeddings_mixed[:, 2], c=labels_mixed, cmap=cmap, norm=norm
    )
    ax2.set_title("NewsEmp24 LLM\n" + r"\textbf{Silhouette score: $\mathbf{" + f"{score_mixed}" + r"}$}")

    ax3 = fig.add_subplot(133, projection="3d")
    score_add = _tsne3d_metrics(embeddings_add_model, labels_add)
    print(f"Silhouette score (add): {score_add}")

    _ = ax3.scatter(
        embeddings_add[:, 0], embeddings_add[:, 1], embeddings_add[:, 2], c=labels_add, cmap=cmap, norm=norm
    )
    ax3.set_title("NewsEmp24 Crowdsourced + NewsEmp22 LLM\n" + r"\textbf{Silhouette score: $\mathbf{" + f"{score_add}" + r"}$}")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")

    cbar = fig.colorbar(sc_base, ax=[ax1, ax2, ax3], aspect=50, pad=0.07, location="right", shrink=0.7)
    cbar.set_label("Empathy score")

    # plt.savefig("logs/tsne/tsne-3d.pdf")

def _plot_tsne2d(config, file_base: list, file_mixed: list, ckpt_path_base: str, ckpt_path_mixed: str):
    embeddings_base, labels_base = _get_embeddings(config, file_base, ckpt_path_base, n_tsne=2)

    config.label_column = config.llm_column
    embeddings_mixed, labels_mixed = _get_embeddings(config, file_mixed, ckpt_path_mixed, n_tsne=2)

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)

    norm = plt.Normalize(1, 7)
    cmap = plt.colormaps["managua"]

    ax1 = fig.add_subplot(121)

    sc_base = ax1.scatter(
        embeddings_base[:, 0], embeddings_base[:, 1], c=labels_base.numpy(), cmap=cmap, norm=norm
    )
    ax1.set_title("Crowdsourced labels")

    ax2 = fig.add_subplot(122)
    sc_mixed = ax2.scatter(
        embeddings_mixed[:, 0], embeddings_mixed[:, 1], c=labels_mixed.numpy(), cmap=cmap, norm=norm
    )
    ax2.set_title("LLM labels")

    for ax in [ax1, ax2]:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    cbar = fig.colorbar(sc_base, ax=[ax1, ax2], orientation="horizontal", aspect=50, pad=0.03, location="bottom", shrink=0.8)
    cbar.set_label("Empathy score")

    plt.savefig("logs/tsne-2d-2-3.pdf", bbox_inches="tight")

def _variance_of_clusters(embd, label):
    # on a second thought, this doesn't make sense because embd is basically a point in the 3D space, so, their variance may not reflect the variance of the clusters
    n_clusters = 6
    bins = np.linspace(1.0, 7.00001, num=n_clusters+1)
    bin_indices = np.digitize(label, bins)
    varianceces = np.zeros(n_clusters)
    for b in np.unique(bin_indices):
        idxs = (bin_indices == b)
        varianceces[b-1] = np.std(embd[idxs], axis=0).mean()
    # print(f"Variance of each cluster: {varianceces}")
    return varianceces.mean()

def _tsne3d_metrics(embd, label):
    n_clusters = 6 # [1.0, 2.0), .[)
    # Convert continuous labels to discrete clusters by binning
    bins = np.linspace(1.0, 7 + 1e-6, num=n_clusters+1) # +1e-6 to avoid the last bin being only 7.0
    label_cluster = np.digitize(label, bins)
    score = silhouette_score(embd, label_cluster, sample_size=1000, random_state=0) # base and mixed has 1000, so sampling this across all cases
    return round(score, 3)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    config = OmegaConf.load("config/config_common.yaml")

    file_base = [config[2024].train]
    ckpt_path_base = "logs/20241115_004656_y(2024)-ImprovedEarlyStop/lr_3e-05_bs_16/seed_0/lightning_logs/version_18332572/checkpoints/epoch=9-step=630.ckpt"
    
    # file_mixed = [config[2024].train_llama, config[2023].train_llama, config[2022].train_llama]
    file_mixed = [config[2024].train_gpt]
    # ckpt_path_mixed = "logs/20241115_003944_y'(2024,2023,2022)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_4.5/seed_42/lightning_logs/version_18332483/checkpoints/epoch=10-step=2849.ckpt"
    # ckpt_path_mixed = "logs/20241115_004406_y'(2024)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_0.0/seed_100/lightning_logs/version_18332544/checkpoints/epoch=7-step=504.ckpt"
    ckpt_path_mixed = "logs/20241230_090054_y'(2024)-gpt/lr_3e-05_bs_16/alpha_0.0/seed_999/lightning_logs/version_19848616/checkpoints/epoch=7-step=504.ckpt"

    ckpt_path_add = "logs/20241116_021714_y(2024)-y_llm(2022)-llm-portion-1.0/lr_3e-05_bs_16/seed_0/lightning_logs/version_18362298/checkpoints/epoch=9-step=1960.ckpt"

    config.seed = 0
    config.extra_columns_to_keep_train = []
    config.train_file_only_LLM_list = []
    config.batch_size = 16

    _plot_tsne3d(config, file_base, file_mixed, ckpt_path_base, ckpt_path_mixed, ckpt_path_add)
