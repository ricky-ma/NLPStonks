import os
import numpy as np
import pandas as pd
import praw
import yfinance as yf
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA

load_dotenv(dotenv_path=Path("..") / ".env")
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)


def predict():
    n_comp = 20
    n_headlines = 22
    news = []
    for submission in reddit.subreddit("WorldNews").top(time_filter='day', limit=n_headlines):
        news.append(submission.title)

    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = np.array(embedder(news))
    pca = PCA(n_components=n_comp)
    embeddings = pca.fit_transform(X=embeddings)

    dija = yf.Ticker('^DJI')
    week = dija.history(period='5d')
    data = pd.DataFrame([week['Close'].tolist()], columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5'])

    for i in range(n_headlines):
        for c in range(n_comp):
            column = 'Top{}_{}'.format(i + 1, c + 1)
            data[column] = [embeddings[i, c]]

    loaded_model = tf.keras.models.load_model('tmp/model')
    pred = loaded_model.predict(data)
    return pred[0][0]

predict()
