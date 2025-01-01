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

    age_bins = [18, 24, 30, 40, 50, 60, 100]
    age_labels =  ["18-24", "25-30", "31-40", "41-50", "51-60", "61+"]
    data["age_group"] = pd.cut(data["age"], bins=age_bins, labels=age_labels, right=False)

    income_bins = [0, 50000, 100000, 150000, 200000, 1000000]
    income_labels = ["0-50K", "50-100K", "100-150K", "150-200K", "200K+"]
    data["income_group"] = pd.cut(data["income"], bins=income_bins, labels=income_labels, right=False)

    return data, age_labels, income_labels

def plt_dmg_dist_single(
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
        if path.endswith(".tsv"):
            data = pd.read_csv(path, sep="\t")
        else:
            raise NotImplementedError("Only TSV files are supported. For CSV, we can use utils/read_file")

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

    if requires_demog_mapping:
        data, _, _ = _demog_mapping_2024(data, demog_columns)

    assert set(demog_columns).issubset(data.columns), f"Some/all demographics columns {demog_columns} not found in the data"

    # Calculate correlations and sample sizes for each race
    group_stats = []
    for feature_value, group in data.groupby(feature):
        sample_size = len(group)
        if sample_size > 1:
            # Calculate Pearson correlation only if the group has more than 1 sample
            correlation, p_value = pearsonr(group['empathy'].to_numpy(), group['pred'].to_numpy())
            ccc = concordance_corrcoef(torch.tensor(group['empathy'].to_numpy()), torch.tensor(group['pred'].to_numpy())).item()
            rmse = mean_squared_error(torch.tensor(group['empathy'].to_numpy()), torch.tensor(group['pred'].to_numpy()), squared=False).item()
        else:
            # Set correlation to NaN if there's only one sample in the group
            correlation, p_value = np.nan, np.nan
            rmse = np.nan
        group_stats.append({feature: feature_value, 'correlation': correlation, 'p_value': p_value, 'sample_size': sample_size, 'rmse': rmse, 'ccc': ccc})

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
                group_stats.loc[len(group_stats)] = {key: value, "correlation": np.nan, "sample_size": 0, "rmse": np.nan, "ccc": np.nan}

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

def subplts_dmg_dist_bar(mode="all"):
    if mode == "train":
        data = pd.read_csv("data/NewsEmp2024/trac3_EMP_train_llama.tsv", sep="\t")
        data = data.rename(columns={"llm_empathy": "pred"})
    elif mode == "pred":
        data = pd.read_csv("data/NewsEmp2024/test_data_with_labels/goldstandard_EMP.csv")
        tst_pred = pd.read_csv("logs/20241115_003944_y'(2024,2023,2022)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_4.5/test-predictions_EMP.tsv", sep="\t", header=None)
        data["pred"] = tst_pred[0]
        data = data.rename(columns={"person_empathy": "empathy"})
    elif mode == "all":
        train = pd.read_csv("data/NewsEmp2024/trac3_EMP_train_llama.tsv", sep="\t")
        dev = pd.read_csv("data/NewsEmp2024/trac3_EMP_dev_llama.tsv", sep="\t")
        test = pd.read_csv("data/NewsEmp2024/test_data_with_labels/goldstandard_EMP_llama.tsv", sep="\t")
        common_cols = train.columns.intersection(dev.columns).intersection(test.columns)
        # print(f"Common columns: {common_cols}")
        data = pd.concat([train[common_cols], dev[common_cols], test[common_cols]], ignore_index=True)
        data = data.rename(columns={"llm_empathy": "pred"})

    continous_features = ["age", "income"]
    categorical_features = ["gender", "education", "race"]
    demog_columns = continous_features + categorical_features
    new_grp = ["age_group", "income_group"]

    data, age_labels, income_labels = _demog_mapping_2024(data, demog_columns)

    possible_categories = {
        'gender': [1, 2, 5],
        'education': [1, 2, 3, 4, 5, 6, 7],
        'race': [1, 2, 3, 4, 5, 6],
        'age_group': age_labels,
        'income_group': income_labels
    }

    feature_labels_map = {
        "gender": {1: "Male", 2: "Female", 5: "Other"},
        "education": {
            1: "Less than high school", 2: "High school", 3: "Technical/Vocational",
            4: "Some college but no degree", 5: "2-year assoc. degree", 6: "4-year bachelor's degree", 7: "Postgraduate/Professional degree"
        },
        "race": {
            1: "White", 2: "Hispanic/Latino", 3: "Black / African American",
            4: "Native American / American Indian", 5: "Asian/Pacific Islander", 6: "Other"
        }
    }

    fig, axes = plt.subplots(1, 5, figsize=(10, 3), sharey=False)
    fig.tight_layout()
    colourmap = cm.viridis_r

    # norm = mcolors.Normalize(vmin=-0.5, vmax=1.0)
    selected_features = categorical_features + new_grp
    for idx, feature in enumerate(selected_features):
        grp_stats = []
        for feature_value, group in data.groupby(feature):
            sample_size = len(group)
            if sample_size > 1:
                # Calculate metrics only if the group has more than 1 sample
                ccc = concordance_corrcoef(torch.tensor(group['empathy'].to_numpy()), torch.tensor(group['pred'].to_numpy())).item()
            else:
                # Set metrics to NaN if there's only one sample in the group
                ccc = np.nan
            grp_stats.append({"category": feature_value, 'ccc': ccc, 'sample_size': sample_size})
 
        grp_stats = pd.DataFrame(grp_stats)
        
        # add missing
        for value in possible_categories[feature]:
            if value not in grp_stats["category"].values:
                grp_stats.loc[len(grp_stats)] = {"category": value, "ccc": np.nan, "sample_size": 0}

        if feature not in new_grp:
            grp_stats["category"] = grp_stats["category"].map(feature_labels_map[feature])

        norm = mcolors.Normalize(vmin=0, vmax=grp_stats["sample_size"].max())
        colours = colourmap(norm(grp_stats["sample_size"]))

        # colour bar based on CCC
        # colours = colourmap(norm(grp_stats["ccc"]))

        bars = axes[idx].bar(
            grp_stats["category"], 
            grp_stats["ccc"].fillna(0), # having empty bar for NaN values
            color=colours, 
            alpha=0.7
        )

        # TODO: make the labels bold
        if feature == "age_group":
            axes[idx].set_xlabel("Age (Years)")
        elif feature == "income_group":
            axes[idx].set_xlabel("Income (USD)")
        else:
            axes[idx].set_xlabel(f"{feature.capitalize()}")
        axes[idx].set_ylabel('CCC')
        axes[idx].set_xticks(grp_stats["category"])
        axes[idx].set_xticklabels(grp_stats["category"], rotation=38, ha="right")

        ymin, ymax = axes[idx].get_ylim()
        axes[idx].set_ylim(ymin, ymax + 0.1)

        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[idx].tick_params(which="both", top=False, right=False)

        # axes[idx].bar(grp_stats["category"], grp_stats["sample_size"], color=colours, alpha=0.7)
        # axes[idx].set_xlabel(f"{feature.capitalize()}")
        # axes[idx].set_ylabel('Sample Size')
        # axes[idx].set_xticks(grp_stats["category"])
        # axes[idx].set_xticklabels([feature_labels_map[feature][value] for value in grp_stats["category"]], rotation=30, ha="right")

        # sm = cm.ScalarMappable(cmap=colourmap, norm=norm)
        # sm.set_array([])
        # cbar = fig.colorbar(sm, ax=axes[idx], orientation='vertical', pad=0.01)
        # cbar.ax.tick_params(labelsize=0, length=0)
        # cbar.set_label('Sample Size', loc="top", labelpad=1)

        # Add text on each bar
        for bar, sample_size in zip(bars, grp_stats["sample_size"]):
            height = max(0, bar.get_height())

            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, # X position, centre of the bar
                height + 0.01, # Y possition; slightly above the bar
                f"N={sample_size}",
                ha="center",
                fontsize=10,
                rotation=90
            )

            # if sample_size <= 1:
            #     ccc_text = "N/A"
            # else:
            #     ccc_text = f"{height:.2f}"

            # axes[idx].text(
            #     bar.get_x() + bar.get_width() / 2,
            #     height,
            #     ccc_text,
            #     ha="center",
            #     fontsize=6
            # )


        # Set x-axis labels
        # labels = [feature_labels_map[feature][value] for value in grp_stats["category"]]
        # axes[idx].set_xticks(range(len(labels)))
        # axes[idx].set_xticklabels(labels, rotation=20, ha="right")

    # Add single colorbar for all subplots
    sm = cm.ScalarMappable(cmap=colourmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, aspect=50, location="top", shrink=0.8)
    # cbar.ax.tick_params(labelsize=0, length=0)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(["0", "Maximum Number of Samples (N)"])
    cbar.ax.set_label('Sample Size')

    plt.savefig(os.path.join("logs/", f"demographic_distribution_{mode}.pdf"), bbox_inches='tight')

def subplts_dmg_dist_scatter():
    data = pd.read_csv("data/NewsEmp2024/trac3_EMP_train_llama.tsv", sep="\t")
    
    continous_features = ["age", "income"]
    categorical_features = ["gender", "education", "race"]
    demog_columns = continous_features + categorical_features

    data, _, _ = _demog_mapping_2024(data, demog_columns)

    possible_categories = {
        'gender': [1, 2, 5],
        'education': [1, 2, 3, 4, 5, 6, 7],
        'race': [1, 2, 3, 4, 5, 6]
    }

    feature_labels_map = {
        "gender": {1: "Male", 2: "Female", 5: "Other"},
        "education": {
            1: "Less than high school", 2: "High school", 3: "Technical/Vocational",
            4: "Some college but no degree", 5: "2-year assoc. degree", 6: "4-year bachelor's degree", 7: "Postgraduate/Professional degree"
        },
        "race": {
            1: "White", 2: "Hispanic/Latino", 3: "Black / African American",
            4: "Native American / American Indian", 5: "Asian/Pacific Islander", 6: "Other"
        }
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

    for idx, feature in enumerate(categorical_features):
        grp_stats = []
        for feature_value, group in data.groupby(feature):
            sample_size = len(group)
            if sample_size > 1:
                # Calculate metrics only if the group has more than 1 sample
                ccc = concordance_corrcoef(torch.tensor(group['empathy'].to_numpy()), torch.tensor(group['llm_empathy'].to_numpy())).item()
            else:
                # Set metrics to NaN if there's only one sample in the group
                ccc = np.nan
            grp_stats.append({"category": feature_value, 'ccc': ccc, 'sample_size': sample_size})
 
        grp_stats = pd.DataFrame(grp_stats)
        
        # add missing
        for value in possible_categories[feature]:
            if value not in grp_stats["category"].values:
                grp_stats.loc[len(grp_stats)] = {"category": value, "ccc": np.nan, "sample_size": 0}

    
        # Map feature values to labels
        grp_stats["category_label"] = grp_stats["category"].map(feature_labels_map[feature])

        # Scatter plot with bubble size proportional to sample size
        scatter = axes[idx].scatter(
            grp_stats["category_label"],
            grp_stats["ccc"],
            s=grp_stats["sample_size"] * 5,  # Adjust scaling factor for bubble size
        )

        axes[idx].set_xlabel(feature.capitalize(), fontsize=12)
        axes[idx].tick_params(axis="x", rotation=45)

        if idx == 0:
            axes[idx].set_ylabel("CCC", fontsize=12)

    plt.savefig(os.path.join("logs/", f"demographic_distribution_scatter.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    # path_2024 = "data/NewsEmp2024/trac3_EMP_train_llama.tsv"
    # plt_dmg_dist_single(path=path_2024, mode="llm_vs_crowd")

    # path_2024 = "data/NewsEmp2024/trac3_EMP_train_llama.tsv"
    # pred_path = "logs/20241106_203926_y'(2024,2023,2022)-SelectedHparam/seed_42/test-predictions_EMP.tsv" 

    # df = annotation_consistency()
    # plot_demographic_distribution(loaded_data=df, requires_demog_mapping=False, feature="education", mode="between_llm")


    subplts_dmg_dist_bar()
    # subplts_dmg_dist_scatter()
    # plt_dmg_dist_single(
    #     path="data/NewsEmp2024/trac3_EMP_train_llama.tsv",
    #     mode="llm_vs_crowd",
    #     requires_demog_mapping=True,
    #     metrics="ccc"
    # )
