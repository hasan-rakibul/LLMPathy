import pandas as pd

def main():
    df = pd.read_csv("data/NewsEmp2024/trac3_EMP_train_llama.tsv", sep="\t")
    df["label_diff"] = abs(df["empathy"] - df["llm_empathy"])
    
    df_sorted = df.sort_values("label_diff", ascending=False)
    
    save_as = "logs/top_differences_NewsEmp24-train.txt"
    with open(save_as, "w") as f:
        f.write("Sorted by largest absolute differences:\n")
        f.write("=" * 80 + "\n")
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"{i}. {row['essay']}\n")
            f.write(f"   Human Empathy: {row['empathy']:.1f}\n")
            f.write(f"   LLM Empathy: {row['llm_empathy']:.1f}\n")
            f.write(f"   Absolute Difference: {row['label_diff']:.1f}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Results saved as '{save_as}'")

if __name__ == "__main__":
    main()
