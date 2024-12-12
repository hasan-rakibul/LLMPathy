import os
import pandas as pd
import glob

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'tableau-colorblind10'])

def plot_llm_portion_sweep(parent_dir: str) -> None:
    glob_str = f"{parent_dir}/**/**-llm-portion-**/**/results.csv"
    result_csvs = glob.glob(glob_str, recursive=True)

    # remove finetunes
    result_csvs = [csv for csv in result_csvs if "finetune" not in csv]

    result_csvs = sorted(result_csvs, key=lambda x: float(x.split("/")[-3].split("-")[-1]))

    all_df = pd.DataFrame()
    for csv in result_csvs:
        df = pd.read_csv(csv, index_col=0)
        df["new_samples"] = int(float(csv.split("/")[-3].split("-")[-1]) * 100)
        all_df = pd.concat([all_df, df])

    mean_df = all_df[all_df.index.str.contains("mean")]
    std_df = all_df[all_df.index.str.contains("std")]
    median_df = all_df[all_df.index.str.contains("median")]

    mean_df = mean_df.set_index("new_samples")
    std_df = std_df.set_index("new_samples")
    median_df = median_df.set_index("new_samples")
    _, ax = plt.subplots(figsize=(5, 3))

    lines = []
    x = median_df.index
    for col in median_df.columns:
        y = median_df[col]
        # std = std_df[col]
        if col in ["val_pcc", "val_ccc"]:
            lines.append(ax.plot(x, y, marker=".", label=col))
            # Plot the shaded area representing the standard deviations
            # _ = ax.fill_between(x, y - std, y + std, alpha=0.1)
        # else:
        #     ax2 = ax.twinx()
        #     lines.append(ax2.plot(x, y, '-', label=col, color='green'))
            # _ = ax2.fill_between(x, y - std, y + std, color='green', alpha=0.1)

    # Adding legend all in one box
    # lines = lines[0] + lines[1] + lines[2]
    lines = lines[0] + lines[1]
    labels = ["l.get_label() for l in lines"]
    labels = [label.split("_")[-1].upper() for label in mean_df.columns] # Extracting the metric name
    ax.legend(lines, labels)

    # annotate the baseline score
    baseline_pcc = 0.331
    baseline_ccc = 0.307
    ax.axhline(y=baseline_pcc, color=lines[0].get_color(), linestyle='--')
    ax.text(50, baseline_pcc+0.007, 'Baseline PCC', color=lines[0].get_color(), fontsize=9, horizontalalignment='center', verticalalignment='center')

    ax.axhline(y=baseline_ccc, color=lines[1].get_color(), linestyle='--')
    ax.text(50, baseline_ccc+0.007, 'Baseline CCC', color=lines[1].get_color(), fontsize=9, horizontalalignment='center', verticalalignment='center')

    # Adding labels and title
    ax.set_xlabel(r'\% New Samples Labelled by LLM')
    ax.set_xticks(x)
    ax.set_ylabel(r'Median Score')
    # ax2.set_ylabel(r'RMSE (Mean $\pm$ SD)')

    # ax.axvline(x=0, color='green', linestyle='--')
    # ax.text(0.03, 0.5, 'LLM', color='green', rotation=90, fontsize=9, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # ax.axvline(x=len(mean_df) - 1, color='red', linestyle='--')
    # ax.text(0.975, 0.5, 'Crowdsourced', color='red', rotation=90, fontsize=10, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)

    plt.savefig(os.path.join(parent_dir, "metrics-vs-llm-portion.pdf"), bbox_inches='tight')
    # plt.close()



if __name__ == "__main__":
    plot_llm_portion_sweep(parent_dir="logs")
