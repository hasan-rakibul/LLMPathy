import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Pool, Manager
from wordcloud import WordCloud # https://github.com/amueller/word_cloud

def text2cloud(text, save_as):
    wordcloud = WordCloud(random_state=42, colormap="binary").generate(text)
    # print(wordcloud.words_) # this is consistent across runs
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("./data/wordcloud/" + save_as, bbox_inches="tight")

def make_train(index, row, train_df, ssl_threshold, q):
    filename = str(index) + "_" + str(row["article_id"]) + "_" + str(row["speaker_id"]) + ".png"
    
    if np.abs(row["crowdsourced_empathy"] - row["gpt_empathy"]) < ssl_threshold:
        # labelled data -- good samples
        q.put([
            filename, 
            round(row[["crowdsourced_empathy", "gpt_empathy"]].mean(), 2),
            "TRAIN",
            "LABELED"
        ])
    else:
        # unlabelled data
        q.put([
            filename,
            round(row["crowdsourced_empathy"], 2),
            "TRAIN",
            "UNLABELED"
        ])
    
    if not os.path.exists("./data/wordcloud/" + filename):
        text2cloud(row["demographic_essay"], filename)

    print("Progress (training): ", round(index/len(train_df)*100), "%", end="\r", flush=True)

def make_valid(index, row, val_df, ssl_threshold, q):
    filename = str(index) + "_" + str(row["article_id"]) + "_" + str(row["speaker_id"]) + ".png"

    q.put([
        filename, 
        round(row["crowdsourced_empathy"], 2), 
        "VAL", 
        "UNLABELED"
    ])

    if not os.path.exists("./data/wordcloud/" + filename):
        text2cloud(row["demographic_essay"], filename)

    print("Progress (validation): ", round(index/len(val_df)*100), "%", end="\r", flush=True)

def main():
    ssl_threshold = 0.5
    train_df = pd.read_csv("./data/v2_v3_train_augmented.tsv", sep="\t")
    val_df = pd.read_csv("./data/v2_dev.tsv", sep="\t")

    file_list = []

    with Manager() as manager:
        file_list_queue = manager.Queue()

        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(
                make_train, 
                [(index, row, train_df, ssl_threshold, file_list_queue) 
                for index, row in train_df.iterrows()]
            )

        while not file_list_queue.empty():
            file_list.append(file_list_queue.get())

    with Manager() as manager:
        file_list_queue = manager.Queue()
        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(
                make_valid, 
                [(index, row, val_df, ssl_threshold, file_list_queue) 
                for index, row in val_df.iterrows()]
            )

        while not file_list_queue.empty():
            file_list.append(file_list_queue.get())

    file_df = pd.DataFrame(file_list, columns=["FileName", "empathy", "SPLIT", "SSL_SPLIT"])
    file_df.to_csv("./data/FileList_SSL_" + str(ssl_threshold) + ".csv", index=False)


if __name__ == "__main__":
    main()
