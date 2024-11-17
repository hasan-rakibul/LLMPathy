import numpy as np
import pandas as pd
import warnings

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

def _extract_test_stat(file: str) -> np.ndarray:
    df = pd.read_csv(file, index_col=0)
    df = df.drop(["mean", "std", "median"], axis=0)
    pcc = df["test_pcc"].to_numpy()
    return pcc

if __name__ == "__main__":
    file_x = "logs/20241114_215420_y(2024,2023,2022)-ImprovedEarlyStop/results.csv"
    file_y = "logs/20241115_003944_y'(2024,2023,2022)-ImprovedEarlyStop-MultiAlpha/lr_3e-05_bs_16/alpha_4.0/results_test.csv"
    pcc_x, pcc_y = _extract_test_stat(file_x), _extract_test_stat(file_y)
    print(f"Expt x: {pcc_x} with mean {np.mean(pcc_x)} and median {np.median(pcc_x)}")
    print(f"Expt y: {pcc_y} with mean {np.mean(pcc_y)} and median {np.median(pcc_y)}")
    # _permutation_test(pcc_x, pcc_y, type="mean")
    # _permutation_test(pcc_x, pcc_y, type="median")
    _t_test(pcc_x, pcc_y)
