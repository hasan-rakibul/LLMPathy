import numpy as np
import pandas as pd

def _permutation_test(data_x: np.ndarray, data_y: np.ndarray) -> None:
    from scipy.stats import permutation_test
    def _mean_diff(x, y, axis=0):
        return np.mean(y - x, axis=axis)
    
    result = permutation_test(
        data=(data_x, data_y),
        statistic=_mean_diff,
        n_resamples=10000, # Number of permutations
        alternative='greater',
        permutation_type='pairings',
        vectorized=True, # requires axis in the statistic function
        axis=0
    )

    print(f"Observed mean difference: {round(result.statistic,3)}")
    print(f"p-value: {result.pvalue}")

def _t_test(data_x: np.ndarray, data_y: np.ndarray) -> None:
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(data_x, data_y)
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_val}")

def _extract_test_stat(file: str) -> np.ndarray:
    df = pd.read_csv(file, index_col=0)
    df = df.drop(["mean", "std", "median"], axis=0)
    pcc = df["test_pcc"].to_numpy()
    return pcc 

if __name__ == "__main__":
    file_x = "logs/20241106_200319_y(2024,2023,2022)-SelectedHparam/results.csv"
    file_y = "logs/20241106_203926_y'(2024,2023,2022)-SelectedHparam/results.csv"
    pcc_x, pcc_y = _extract_test_stat(file_x), _extract_test_stat(file_y)
    print(f"Expt x: {pcc_x} with mean {np.mean(pcc_x)}")
    print(f"Expt y {pcc_y} with mean {np.mean(pcc_y)}")
    # _permutation_test(pcc_x, pcc_y)
    _t_test(pcc_x, pcc_y)
