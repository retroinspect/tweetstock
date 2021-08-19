"""
Label sentiment to NASDAQ tweets
"""
import pandas as pd
import numpy as np
from fintweet_sentiment_classification_model import BertweetForSequenceClassification
from utils import tokenize_function, save
from datasets import Dataset
from transformers import Trainer

MENTIONED_COMPANY_FILEPATH = "./data/raw/company-tweets.csv"
TWEETS_FILEPATH = "./data/regression-tweets-raw.csv"
# MENTIONED_COMPANY_FILEPATH = "./data/sample/company-tweets.csv"
# TWEETS_FILEPATH = "./data/sample/regression-tweets-raw.csv"

DELIM = ','
COL_NEEDED=['sentiment_level', 'datetime']

# company_mentioning_tweet_id: tweet_id,ticker_symbol
# tweets: id,datetime,body
company_mentioning_tweet_id = pd.read_csv(MENTIONED_COMPANY_FILEPATH, engine='python', delimiter=DELIM)
tweets = pd.read_csv(TWEETS_FILEPATH, engine='python', delimiter=DELIM)

# select MAX_NUM_TWEETS tweets that only mentions one company at a time
company_mentioning_tweet_id = company_mentioning_tweet_id.drop_duplicates('tweet_id')
# company_mentioning_tweet_id = company_mentioning_tweet_id.iloc[:MAX_NUM_TWEETS]

# tag sentiment for all the tweets
# sentiment_labeled_tweets: datetime,sentiment_level,ticker_symbol
# sentiment_level: 1(positive), 0(neutral), -1(negative)
sentiment_classifier = BertweetForSequenceClassification.from_pretrained("./fintweet-sentiment-classifier", num_labels=3)
raw_datasets = Dataset.from_pandas(tweets)
tokenized_tweets = raw_datasets.map(tokenize_function, batched=True)
trainer = Trainer(model=sentiment_classifier)
logits, _, _ = trainer.predict(test_dataset=tokenized_tweets)
tweets['sentiment_level'] = np.argmax(logits, axis=-1)-1
tweets = tweets.rename(columns={'id' : 'tweet_id'})
tweets = tweets.join(company_mentioning_tweet_id.set_index('tweet_id'), on='tweet_id', how='left')
tweets.dropna(subset=['ticker_symbol'], inplace=True)

# split by company and save
companies = ['AAPL', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'MSFT']
for ticker in companies:
  company_tweets = tweets[tweets['ticker_symbol'] == ticker]
  company_tweets.drop(company_tweets.columns.difference(COL_NEEDED), axis=1, inplace=True)

  print(f"{ticker} tweets: {len(company_tweets)}")
  if (len(company_tweets) > 0):
    save(company_tweets, f"nasdaq-tweet-sentiment-{ticker}", "./data/company-sentiment")
