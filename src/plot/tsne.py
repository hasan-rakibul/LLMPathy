import os
import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
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

def _get_embeddings(config, filepath, ckpt_path):
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
        perplexity=30,
        random_state=config.seed,
        # method="exact",
        n_jobs=-1,
        verbose=1
    )
    tsne_embeddings = tsne.fit_transform(embeddings.numpy())

    return tsne_embeddings, labels

def _plot_tsne(config, file_base: list, file_mixed: list, ckpt_path_base: str, ckpt_path_mixed: str):
    embeddings_base, labels_base = _get_embeddings(config, file_base, ckpt_path_base)

    config.label_column = config.llm_column
    embeddings_mixed, labels_mixed = _get_embeddings(config, file_mixed, ckpt_path_mixed)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    norm = plt.Normalize(1, 7)
    cmap = plt.colormaps["managua"]

    sc_base = axs[0].scatter(
        embeddings_base[:, 0], embeddings_base[:, 1], c=labels_base.numpy(), cmap=cmap, norm=norm
    )
    axs[0].set_title("Crowdsourced labels")

    sc_mixed = axs[1].scatter(
        embeddings_mixed[:, 0], embeddings_mixed[:, 1], c=labels_mixed.numpy(), cmap=cmap, norm=norm
    )
    axs[1].set_title("LLM labels")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(sc_base, ax=axs, orientation="horizontal", aspect=50, pad=0.03, location="top", shrink=0.8)
    cbar.set_label("Empathy score")

    plt.savefig("logs/tsne.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    config = OmegaConf.load("config/config_common.yaml")

    file_base = [config[2024].train]
    ckpt_path_base = "logs/20241115_004656_y(2024)-ImprovedEarlyStop/lr_3e-05_bs_16/seed_0/lightning_logs/version_18332572/checkpoints/epoch=9-step=630.ckpt"
    
    # file_mixed = [config[2024].train_llama, config[2023].train_llama, config[2022].train_llama]
    file_mixed = [config[2024].train_llama]
    # ckpt_path_mixed = "logs/20241115_003944_y'(2024,2023,2022)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_4.5/seed_42/lightning_logs/version_18332483/checkpoints/epoch=10-step=2849.ckpt"
    ckpt_path_mixed = "logs/20241115_004406_y'(2024)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_0.0/seed_100/lightning_logs/version_18332544/checkpoints/epoch=7-step=504.ckpt"

    config.seed = 0
    config.extra_columns_to_keep_train = []
    config.train_file_only_LLM_list = []
    config.batch_size = 16

    _plot_tsne(config, file_base, file_mixed, ckpt_path_base, ckpt_path_mixed)