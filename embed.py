import numpy as np
import pandas as pd
import tensorflow_hub as hub
from bert_serving.client import BertClient


def embed(column):
    bc = BertClient()
    news_dija_df = pd.read_csv("data/Combined_News_DJIA.csv")
    headlines = news_dija_df[column].tolist()
    embeddings = bc.encode(headlines)
    with open('embeddings/embeddings_{}.npy'.format(column), 'wb') as f:
        np.save(f, embeddings)


def embed_USE(column):
    news_dija_df = pd.read_csv("data/Combined_News_DJIA.csv")
    headlines = news_dija_df[column].tolist()
    embeddings = np.array(embedder(headlines))
    with open('embeddings_USE/embeddings_{}.npy'.format(column), 'wb') as f:
        np.save(f, embeddings)


# bert-serving-start -model_dir tmp/uncased_L-24_H-1024_A-16/ -num_worker=1 -cpu -max_seq_len=NONE
if __name__ == "__main__":
    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    for col in range(1, 26):
        print("Embedding column {}".format(col))
        embed_USE("Top{}".format(col))
