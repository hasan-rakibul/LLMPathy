import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from statannotations.Annotator import Annotator
plt.style.use(['science', 'tableau-colorblind10'])

def _permutation_test(data_x: np.ndarray, data_y: np.ndarray, type: str = "mean") -> None:
    warnings.warn("I suspect there is something wrong with my implementation of permutation test")
    from scipy.stats import permutation_test
    def _mean_diff(x, y, axis=0):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)
    
    def _median_diff(x, y, axis=0):
        return np.median(y, axis=axis) - np.median(x, axis=axis)
    
    if type == "mean":
        statistic = _mean_diff
    elif type == "median":
        statistic = _median_diff
    else:
        raise ValueError("Invalid type")
    
    result = permutation_test(
        data=(data_x, data_y),
        statistic=statistic,
        n_resamples=10000, # Number of permutations
        alternative='two-sided',
        permutation_type='pairings',
        vectorized=True, # requires axis in the statistic function
        axis=0
    )

    print(f"\nObserved {type} difference: {round(result.statistic,3)}")
    print(f"p-value: {result.pvalue}")

def _t_test(data_x: np.ndarray, data_y: np.ndarray) -> None:
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(data_x, data_y)
    print(f"\nt-statistic: {t_stat}")
    print(f"p-value: {p_val}")

def _extract_test_stat(file: str, metric: str = "pcc") -> np.ndarray:
    df = pd.read_csv(file, index_col=0)
    df = df.drop(["mean", "std", "median"], axis=0)
    metric = df[f"test_{metric}"].to_numpy()
    return metric

def _print_significance(file_x: str, file_y: str, metrics: list) -> None:
    for metric in metrics:
        print(f"\n\n{metric}")
        metric_x, pcc_y = _extract_test_stat(file_x, metric=metric), _extract_test_stat(file_y, metric=metric)
        print(f"Expt x: {metric_x} with mean {np.mean(metric_x)} and median {np.median(metric_x)}")
        print(f"Expt y: {pcc_y} with mean {np.mean(pcc_y)} and median {np.median(pcc_y)}")
        # _permutation_test(pcc_x, pcc_y, type="mean")
        # _permutation_test(pcc_x, pcc_y, type="median")
        _t_test(metric_x, pcc_y)

def _plot_significance(file_x: str, file_y: str, file_z: str, metrics: list) -> None:
    expt_x = "Crowdsourced labels"
    expt_y = "Crowdsourced \& LLM mixed"
    expt_z = "Crowdsourced + LLM-labelled extra data"
    # Prepare data for plotting
    data = []
    for metric in metrics:
        metric_x = _extract_test_stat(file_x, metric=metric)
        metric_y = _extract_test_stat(file_y, metric=metric)
        metric_z = _extract_test_stat(file_z, metric=metric)
        data.extend([
            {"Experiment": expt_x, "Metric": metric.upper(), "Value": val} for val in metric_x
        ])
        data.extend([
            {"Experiment": expt_y, "Metric": metric.upper(), "Value": val} for val in metric_y
        ])
        data.extend([
            {"Experiment": expt_z, "Metric": metric.upper(), "Value": val} for val in metric_z
        ])

    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(data=df, x="Metric", y="Value", hue="Experiment", errorbar="sd")

    # Add significance annotations
    pairs = [
        (("PCC", expt_x), ("PCC", expt_y)),
        (("CCC", expt_x), ("CCC", expt_y)),
        (("RMSE", expt_x), ("RMSE", expt_y)),
        (("PCC", expt_x), ("PCC", expt_z)),
        (("CCC", expt_x), ("CCC", expt_z)),
        (("RMSE", expt_x), ("RMSE", expt_z)),
        # (("PCC", expt_y), ("PCC", expt_z)),
        # (("CCC", expt_y), ("CCC", expt_z)),
        # (("RMSE", expt_y), ("RMSE", expt_z)),
    ]
    annotator = Annotator(ax, pairs, data=df, x="Metric", y="Value", hue="Experiment")
    annotator.configure(test="t-test_ind", text_format="star", loc="inside", verbose=2)
    annotator.apply_and_annotate()

    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metric")
    ax.set_xticklabels([r"PCC $\uparrow$", r"CCC $\uparrow$", r"RMSE $\downarrow$"])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    plt.savefig("logs/significance_plot_mixed_and_extra.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    file_x = "logs/20241115_004656_y(2024)-ImprovedEarlyStop/lr_3e-05_bs_16/results_test.csv"
    # file_y = "logs/20241115_004406_y'(2024)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_3.5/results_test.csv"
    file_y = "logs/20241230_090054_y'(2024)-gpt/lr_3e-05_bs_16/alpha_3.5/results_test.csv"
    # file_z = "logs/20241118_182138_y(2024)-y_llm(2022)-llm-portion-1.0/lr_3e-05_bs_16/results_test.csv"
    file_z = "logs/20241116_021714_y(2024)-y_llm(2022)-llm-portion-1.0/lr_3e-05_bs_16/results_test.csv"
    metrics = ["pcc", "ccc", "rmse"]
    # _print_significance(file_x, file_y, metrics)
    _plot_significance(file_x, file_y, file_z, metrics)
