import pandas as pd
import argparse

def _make_latex_table_row(path: str) -> None:
    df = pd.read_csv(path, index_col=0)
    columns = df.columns

    if "mean" in df.index:
        # if mean is present, then std and median will also be present
        mean_row = df.loc["mean", :].round(3)
        std_row = df.loc["std", :].round(3)
        median_row = df.loc["median", :].round(3)
        results_only = df.drop(["mean", "std", "median"])
    else:
        mean_row = df.mean().round(3)
        std_row = df.std().round(3)
        median_row = df.median().round(3)
        results_only = df
        
    best_scores = pd.DataFrame(index=["best"], columns=columns)
    for col in results_only.columns:
        if col.endswith("_rmse"):
            best_scores.loc["best", col] = df[col].min()
        else:
            best_scores.loc["best", col] = df[col].max()

    best_row = best_scores.loc["best", :].round(3)

    print(" & ".join(columns))

    print("\nMean(±std)")
    print(" & ".join([f"${mean}\\pm {std}$"\
            for mean, std in zip(mean_row, std_row)]))
    
    print("\nMedian")
    print( " & ".join([f"${median}$"\
            for median in median_row]))
    
    print("\nMedian(Best)")
    print(" & ".join([f"${median}({best})$" for median, best in zip(median_row, best_row)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    _make_latex_table_row(args.path)
