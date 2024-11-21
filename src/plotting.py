import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scienceplots
plt.style.use(['science', 'tableau-colorblind10'])

from utils import read_file

def plot(x, y, y2=None, xlabel=None, ylabel=None, legend=[], save=False, filename=None):
    """Plot data points"""
    plt.style.use(['science'])
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(x, y)
    if y2 is not None:
        ax.plot(x, y2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    
    if save:
        plt.savefig(fname=filename+'.pdf', format='pdf', bbox_inches='tight')
        print(f"Saved as {filename}.pdf")
        
    fig.show()

def _demog_mapping_2024(data: pd.DataFrame, demog_columns: list[str]) -> pd.DataFrame:
    # require mapping for 2024 demographics
    # demographics mapping are available from PER dataset
    per_train = pd.read_csv("data/NewsEmp2024/demographic-from-PER-task/trac4_PER_train.csv", index_col=0)
    per_dev = pd.read_csv("data/NewsEmp2024/demographic-from-PER-task/trac4_PER_dev.csv", index_col=0)
    per_test = pd.read_csv("data/NewsEmp2024/demographic-from-PER-task/goldstandard_PER.csv", index_col=None)
    assert per_train.columns.to_list() == per_dev.columns.to_list() == per_test.columns.to_list()
    
    demog_map = pd.concat([per_train, per_dev, per_test], ignore_index=True)
    demog_map.drop_duplicates(subset=['person_id'], inplace=True)
    
    data["person_id"] = data["person_id"].astype(str) # for merging
    demog_map["person_id"] = demog_map["person_id"].astype(str) # for merging

    only_demog_map = demog_map[['person_id'] + demog_columns] # person_id is important for mapping
    data = pd.merge(data, only_demog_map, on='person_id', how='left', validate='many_to_one')
    return data

def plot_demographic_distribution(
        path: str = None,
        pred_path: str = None,
        feature: str = "race",
        mode: str = "with_llm",
        requires_demog_mapping: bool = True,
        loaded_data: pd.DataFrame = None,
        metrics: str = "ccc"
    ) -> None:
    
    if path is None:
        print("Using loaded data")
        data = loaded_data
    else:
        print(f"Reading data from {path}")
        data = read_file(path)

    if mode == "llm_vs_crowd":
        print("llm empathy vs crowedsourced empathy")
        assert "empathy" in data.columns, "empathy column not found in the data"
        data = data.rename(columns={"llm_empathy": "pred"})
    elif mode == "llm_vs_llm":
        print("llm empathy vs llm empathy")
        if "empathy" in data.columns:
            data = data.drop(columns=["empathy"])
        if "pred" in loaded_data.columns:
            data = data.drop(columns=["pred"])
        data = data.rename(columns={"llm_empathy_1": "empathy", "llm_empathy_2": "pred"})
    elif mode == "pred_vs_crowd":
        print("pred empathy vs crowdsourced empathy")
        assert "empathy" in data.columns, "empathy column not found in the data"
        pred = pd.read_csv(pred_path, sep="\t", header=None)
        data["pred"] = pred[0]

    continous_features = ["age", "income"]
    categorical_features = ["gender", "education", "race"]
    demog_columns = continous_features + categorical_features

    if requires_demog_mapping: # volatile check, depending on the dir/file name
        data = _demog_mapping_2024(data, demog_columns)

    assert set(demog_columns).issubset(data.columns), f"Some/all demographics columns {demog_columns} not found in the data"

    # Calculate correlations and sample sizes for each race
    group_stats = []
    for feature_value, group in data.groupby(feature):
        sample_size = len(group)
        if sample_size > 1:
            # Calculate Pearson correlation only if the group has more than 1 sample
            correlation, p_value = pearsonr(group['empathy'].to_numpy(), group['pred'].to_numpy())
            rmse = mean_squared_error(torch.tensor(group['empathy'].to_numpy()), torch.tensor(group['pred'].to_numpy()), squared=False).item()
        else:
            # Set correlation to NaN if there's only one sample in the group
            correlation, p_value = np.nan, np.nan
            rmse = np.nan
        group_stats.append({feature: feature_value, 'correlation': correlation, 'p_value': p_value, 'sample_size': sample_size, 'rmse': rmse})

    group_stats = pd.DataFrame(group_stats)
    possible_categories = {
        'gender': [1, 2, 5],
        'education': [1, 2, 3, 4, 5, 6, 7],
        'race': [1, 2, 3, 4, 5, 6]
    }
    # add missing race
    for key, values in possible_categories.items():
        if key not in group_stats.columns:
            continue
        for value in values:
            if value not in group_stats[key].values:
                group_stats.loc[len(group_stats)] = {key: value, "correlation": np.nan, "sample_size": 0, "rmse": np.nan}

    group_stats = group_stats.sort_values(feature)
    print(group_stats)

    # Normalise the sample_size values to the range [0, 1] for colormap mapping
    norm = mcolors.Normalize(vmin=group_stats["sample_size"].min(), vmax=group_stats["sample_size"].max())
    colourmap = cm.viridis_r
    colours = colourmap(norm(group_stats["sample_size"]))

    # Plotting
    _, ax1 = plt.subplots(figsize=(6, 4))

    ax1.bar(group_stats[feature], group_stats[metrics], color=colours, alpha=0.7, label=metrics.capitalize())
    ax1.set_xlabel(f'{feature.capitalize()}')
    ax1.set_ylabel('RMSE', color='blue')
    ax1.set_xticks(group_stats[feature])
    
    if feature == "race":
        feature_labels = {1: "White", 2: "Hispanic or Latino", 3: "Black or African American", 4: "Native American or American Indian", 5: "Asian/Pacific Islander", 6: "Other"}
    elif feature == "gender":
        feature_labels = {1: "Male", 2: "Female", 5: "Other"}
    elif feature == "education":
        feature_labels = {1: "Less than high school", 2: "High school", 3: "Technical/Vocational", 4: "Some college but no degree", 5: "2-year associate degree", 6: "4-year bachelor's degree", 7: "Postgraduate/professional degree"}
    else:
        raise ValueError(f"Unknown feature {feature}")
    
    ax1.set_xticklabels([feature_labels[key] for key in group_stats[feature]], rotation=20, ha='right')
    ax1.tick_params(axis="y", labelcolor='blue')
    ax1.set_ylim(0, group_stats["rmse"].max() + 0.06)

    # Add a colorbar to indicate sample_size
    sm = cm.ScalarMappable(cmap=colourmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Sample Size')

    # Secondary y-axis for correlation
    # ax2 = ax1.twinx()
    # ax2.scatter(group_stats[feature], group_stats['sample_size'], marker=".", color='red', label='Sample Size', zorder=5)
    # ax2.set_ylabel('Sample Size', color='red')
    # ax2.tick_params(axis="y", labelcolor='red')
    # ax2.set_ylim(0, group_stats["sample_size"].max() + 50)

    # for i, row in group_stats.iterrows():
    #     ax1.text(row[feature], row["sample_size"]+2, f"N = {row['sample_size']:.0f}", ha='center', va='bottom', color='blue')
    #     if row['sample_size'] > 1:
    #         ax2.text(row[feature], row['correlation']+0.01, f"PCC = ${row['correlation']:.2f}$", ha='center', va='bottom', color='red', fontweight='bold')
    #     else:
    #         ax2.text(row[feature], group_stats["correlation"].min()+0.04, "PCC = N/A", ha='center', va='bottom', color='gray', fontstyle='italic')

    # Legends
    # fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
    plt.savefig(os.path.join("logs/", f"distribution_{feature}_{mode}.pdf"), bbox_inches='tight')
    plt.show()

def annotation_consistency() -> pd.DataFrame:
    tr_24 = pd.read_csv("data/NewsEmp2024/trac3_EMP_train_llama.tsv", sep="\t")
    tr_23 = pd.read_csv("data/NewsEmp2023/WASSA23_essay_level_with_labels_train_llama.tsv", sep="\t")
    dv_23 = pd.read_csv("data/NewsEmp2023/WASSA23_essay_level_dev_llama.tsv", sep="\t")

    print(tr_24.shape, tr_23.shape, dv_23.shape)

    common_cols = tr_23.columns.intersection(dv_23.columns)
    tr_23 = tr_23[common_cols]
    dv_23 = dv_23[common_cols]
    tr_dv_23 = pd.concat([tr_23, dv_23], ignore_index=True)
    print(tr_dv_23.shape)

    tr_24 = tr_24.drop_duplicates(subset=["essay"])
    tr_dv_23 = tr_dv_23.drop_duplicates(subset=["essay"])
    print(tr_24.shape, tr_dv_23.shape)

    tr_24 = tr_24.rename(columns={"llm_empathy": "llm_empathy_1"})
    tr_dv_23 = tr_dv_23.rename(columns={"llm_empathy": "llm_empathy_2"})

    merged_df = pd.merge(tr_24, tr_dv_23, on="essay", how="inner", validate="one_to_one")

    common_cols = tr_24.columns.intersection(tr_dv_23.columns)
    for col in common_cols:
        if col == "essay":
            continue

        if merged_df[col + "_x"].equals(merged_df[col + "_y"]):
            print(f"{col} is equal")
            merged_df.drop(col + "_y", axis=1, inplace=True)
            merged_df.rename(columns={col + "_x": col}, inplace=True)
        else:
            print(f"{col} is not equal")

    llm_empathy_1 = torch.tensor(merged_df["llm_empathy_1"].values)
    llm_empathy_2 = torch.tensor(merged_df["llm_empathy_2"].values)

    pcc = pearson_corrcoef(llm_empathy_1, llm_empathy_2).item()
    ccc = concordance_corrcoef(llm_empathy_1, llm_empathy_2).item()
    rmse = mean_squared_error(llm_empathy_1, llm_empathy_2, squared=False).item()
    pcc = round(pcc, 3)
    ccc = round(ccc, 3)
    rmse = round(rmse, 3)

    print(f"PCC: {pcc}, CCC: {ccc}, RMSE: {rmse}")

    merged_df["llm_diff"] = np.abs(merged_df["llm_empathy_1"] - merged_df["llm_empathy_2"])
    mean_diff = merged_df["llm_diff"].mean().round(3)
    std_diff = merged_df["llm_diff"].std().round(3)
    print(f"Difference - Mean: {mean_diff}, Std: {std_diff}")

    return merged_df


if __name__ == "__main__":
    path_2024 = "data/NewsEmp2024/trac3_EMP_train_llama.tsv"
    plot_demographic_distribution(path=path_2024, mode="llm_vs_crowd")

    # path_2024 = "data/NewsEmp2024/trac3_EMP_train_llama.tsv"
    # pred_path = "logs/20241106_203926_y'(2024,2023,2022)-SelectedHparam/seed_42/test-predictions_EMP.tsv" 

    # df = annotation_consistency()
    # plot_demographic_distribution(loaded_data=df, requires_demog_mapping=False, feature="education", mode="between_llm")

