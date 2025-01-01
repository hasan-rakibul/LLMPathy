import os
import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
            output = model(batch)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu())
            labels.append(batch["labels"])

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    tsne = TSNE(
        n_components=n_tsne,
        perplexity=30,
        random_state=config.seed,
        # method="exact",
        n_jobs=-1,
        verbose=1
    )
    tsne_embeddings = tsne.fit_transform(embeddings.numpy())

    return tsne_embeddings, labels

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

    return tsne_embeddings, labels

def _plot_tsne3d(config, file_base: list, file_mixed: list, ckpt_path_base: str, ckpt_path_mixed: str, ckpt_path_add: str):
    embeddings_base, labels_base = _get_embeddings(config, file_base, ckpt_path_base)

    config.label_column = config.llm_column
    embeddings_mixed, labels_mixed = _get_embeddings(config, file_mixed, ckpt_path_mixed)

    config.label_column = "empathy" # reset
    embeddings_add, labels_add = _get_embeddings_additional(config, ckpt_path_add)

    fig = plt.figure(figsize=(14, 6), constrained_layout=False)

    norm = plt.Normalize(1, 7)
    cmap = plt.colormaps["managua"]

    ax1 = fig.add_subplot(131, projection="3d")

    sc_base = ax1.scatter(
        embeddings_base[:, 0], embeddings_base[:, 1], embeddings_base[:, 2], c=labels_base.numpy(), cmap=cmap, norm=norm
    )
    ax1.set_title("NewsEmp24 Crowdsourced")

    ax2 = fig.add_subplot(132, projection="3d")
    sc_mixed = ax2.scatter(
        embeddings_mixed[:, 0], embeddings_mixed[:, 1], embeddings_mixed[:, 2], c=labels_mixed.numpy(), cmap=cmap, norm=norm
    )
    ax2.set_title("NewsEmp24 LLM")

    ax3 = fig.add_subplot(133, projection="3d")
    sc_add = ax3.scatter(
        embeddings_add[:, 0], embeddings_add[:, 1], embeddings_add[:, 2], c=labels_add.numpy(), cmap=cmap, norm=norm
    )
    ax3.set_title("NewsEmp24 Crowdsourced + NewsEmp22 LLM")


    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")

    cbar = fig.colorbar(sc_base, ax=[ax1, ax2, ax3], aspect=50, pad=0.07, location="right", shrink=0.7)
    cbar.set_label("Empathy score")

    plt.savefig("logs/tsne-3d.pdf")

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
    # _plot_tsne2d(config, file_base, file_mixed, ckpt_path_base, ckpt_path_mixed)