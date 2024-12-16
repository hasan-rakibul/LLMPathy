import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'tableau-colorblind10'])

def plot_metrics_vs_alpha(parent_dir: str) -> None:
    glob_str = f"{parent_dir}/**/results.csv"
    result_csvs = glob.glob(glob_str, recursive=True)
    result_csvs = sorted(result_csvs)

    all_df = pd.DataFrame()
    for csv in result_csvs:
        df = pd.read_csv(csv, index_col=0)
        df["alpha"] = csv.split("/")[-2].split("_")[-1]
        all_df = pd.concat([all_df, df])
    
    mean_df = all_df[all_df.index.str.contains("mean")]
    std_df = all_df[all_df.index.str.contains("std")]
    
    mean_df = mean_df.set_index("alpha")
    std_df = std_df.set_index("alpha")
    _, ax = plt.subplots(figsize=(5, 3))

    lines = []
    x = mean_df.index
    for col in mean_df.columns:
        y = mean_df[col]
        std = std_df[col]
        if col in ["val_pcc", "val_ccc"]:
            lines.append(ax.plot(x, y, '.', label=col))
            # Plot the shaded area representing the standard deviations
            _ = ax.fill_between(x, y - std, y + std, alpha=0.1)
        else:
            ax2 = ax.twinx()
            lines.append(ax2.plot(x, y, '.', label=col, color='green'))
            _ = ax2.fill_between(x, y - std, y + std, color='green', alpha=0.1)

    # Adding legend all in one box
    lines = lines[0] + lines[1] + lines[2]
    labels = ["l.get_label() for l in lines"]
    labels = [label.split("_")[-1].upper() for label in mean_df.columns] # Extracting the metric name
    ax.legend(lines, labels, loc=(0.05,0.05))

    # Adding labels and title
    ax.set_xlabel(r'Annotation Selection Threshold ($\alpha$)')
    ax.set_ylabel(r'PCC/CCC Metrics (Mean $\pm$ SD)')
    ax2.set_ylabel(r'RMSE (Mean $\pm$ SD)')

    ax.axvline(x=0, color='green', linestyle='--')
    ax.text(0.03, 0.5, 'LLM', color='green', rotation=90, fontsize=9, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ax.axvline(x=len(mean_df) - 1, color='red', linestyle='--')
    ax.text(0.975, 0.5, 'Crowdsourced', color='red', rotation=90, fontsize=10, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)

    plt.savefig(os.path.join(parent_dir, "val-metrics_vs_alpha.pdf"), bbox_inches='tight')
    print(f"Saved plot to {parent_dir}")
    plt.close()


if __name__ == "__main__":
    plot_metrics_vs_alpha("logs/20241115_004406_y'(2024)-ImprovedEarlyStop-MultiAlpha")