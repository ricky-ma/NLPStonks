from datetime import datetime
import pandas as pd
from inference import load_headlines, predict


headlines_urls = load_headlines(22)
headlines = [x[0] for x in headlines_urls]
pred = predict(headlines)
pred_df = pd.read_csv('data/predictions.csv')
pred_df.loc[len(pred_df)] = [datetime.today().strftime('%Y-%m-%d'), pred]
pred_df.to_csv('data/predictions.csv', index=False)
