import pandas as pd
import numpy as np
from utils import save

MAX_NUM_TWEETS = 30000

nasdaq_tweets = {
  'FILEPATH':"./data/nasdaq-tweets.csv",
  'DELIM':',',
  'DATETIME_TYPE':'UNIX',
  'DROP_IF_NULL':['writer'],
  'COL_NEEDED': ['body', 'datetime', 'id'],
  'DATETIME_FORMAT': None
}

sentiment_labeled1 = {
  'FILEPATH': "./data/sentiment-labeled-financial-tweets.csv",
  'DELIM': ';',
  'DATETIME_FORMAT': '%Y-%m-%d %H:%M:%S+00:00',
  'DATETIME_TYPE': 'ISO',
  'DROP_IF_NULL': ['writer', 'sentiment'],
  'COL_NEEDED': ['body', 'sentiment']
}

sentiment_labeled2 = {
  'FILEPATH':"./data/sentiment-labeled-financial-tweets2.csv",
  'DELIM':',',
  'DATETIME_FORMAT':'%Y-%m-%d %H:%M:%S+00:00',
  'DATETIME_TYPE':'ISO',
  'DROP_IF_NULL':['sentiment'],
  'COL_NEEDED' :['body', 'sentiment']
}

def clean(kwargs):
  FILEPATH = kwargs['FILEPATH']
  DELIM = kwargs['DELIM']  
  DATETIME_FORMAT = kwargs['DATETIME_FORMAT']
  DATETIME_TYPE = kwargs['DATETIME_TYPE']
  DROP_IF_NULL = kwargs['DROP_IF_NULL']
  COL_NEEDED = kwargs['COL_NEEDED']

  print(f"preprocessing {FILEPATH}...")
  print(f"Removing bot tweets from {FILEPATH}")

  tweets = pd.read_csv(FILEPATH, engine='python', delimiter=DELIM)
  print(f"Initial tweets: {len(tweets)}")

  tweets = tweets.dropna(subset=DROP_IF_NULL)
  print(f"Dropped entries of which column is null: now {len(tweets)}")

  tweets['prep_body'] = tweets['body'].replace(r"https?:\S+|http?:\S+|www?:\S+", '', regex=True).replace(r"[@#\$][a-zA-Z]+", '', regex=True).replace(r"\s\s+", ' ', regex=True).str.strip()
  tweets = tweets.drop_duplicates(subset=['prep_body'])
  print(f"Dropped all duplicated entries: now {len(tweets)}")

  # max tweets rate
  if DATETIME_TYPE=='UNIX':
    tweets['datetime'] = pd.to_datetime(tweets['datetime'], unit='s', errors='coerce')
  elif DATETIME_TYPE=='ISO':
    tweets['datetime'] = pd.to_datetime(tweets['datetime'], format=DATETIME_FORMAT, errors='coerce')
  else:
    raise Exception(f"Invalid datetime type: {DATETIME_TYPE} should be one of UNIX or ISO")

  tweets['hour'] = tweets['datetime'].dt.hour
  tweets['date'] = tweets['datetime'].dt.date

  data = tweets[['writer', 'hour', 'date', 'body']].groupby(['writer', 'hour', 'date']).count().reset_index().rename(columns={'body' : 'tweet_rate'})

  indmax = data.groupby('writer').agg({'tweet_rate' : 'idxmax'})
  posters = data.iloc[indmax.tweet_rate].sort_values(by='tweet_rate').set_index('writer')
  posters = posters.drop(['hour', 'date'], axis=1).rename(columns={'tweet_rate' : 'max_tweet_rate'})

  hours = data[['writer', 'hour', 'tweet_rate']].groupby(['writer', 'hour']).mean().sort_values(by='tweet_rate')
  hours = hours.reset_index().pivot(index='writer', columns='hour', values='tweet_rate').fillna(0)
  hours.columns.name = None
  posters = posters.join(hours, how='outer')
  posters.sort_values(by='max_tweet_rate').head()

  for h in range(24):
    if not h in posters:
      posters[h] = 0.0

  # Average time between subsequent tweets
  def in_qrange(ser, q):
      return ser.between(*ser.quantile(q=q))

  tweets['timediff'] = tweets.sort_values('datetime', ascending=False).groupby(['writer']).datetime.diff(-1).dt.seconds.fillna(np.inf)
  data = tweets.loc[tweets['timediff'].transform(in_qrange, q=[0, 0.75]), ['writer', 'timediff']].groupby('writer').agg(['mean']).rename(columns={'mean' : 'mean_diff_sec'})
  data.columns = data.columns.droplevel()

  tweets.drop(['timediff'], inplace=True, axis=1)

  columns = list(range(24))
  bot_check = pd.DataFrame(index=posters.index)

  if len(data['mean_diff_sec']) != 0:
    posters = posters.join(data, on='writer', how='left').fillna(max(data['mean_diff_sec']))
    posters.loc[posters['mean_diff_sec'] == 0, 'mean_diff_sec'] = max(data['mean_diff_sec'])
    posters.sort_values(by='mean_diff_sec').head()
    bot_check["mean_diff_sec"] = (posters["mean_diff_sec"] < 10).astype(np.int8)
    print("mean time between tweets sec < 5 seconds : {} writers".format(sum(bot_check["mean_diff_sec"])))

  bot_check["max_tweet_rate"] = (posters["max_tweet_rate"] > 100).astype(np.int8)
  print("max hourly tweets rate > 100 : {} writers".format(sum(bot_check["max_tweet_rate"])))

  bot_check["abscence_hours"] = ((posters[columns] == 0).astype(int).sum(axis=1) < 3).astype(np.int8)
  print("less than 3 hours of not tweeting : {} writers".format(sum(bot_check["abscence_hours"])))

  bot_tweets = tweets.loc[tweets['writer'].isin(bot_check[bot_check.sum(axis=1) > 1].index), 'prep_body'].unique()
  bot_check['tweet_like_bot'] = bot_check.index.isin(tweets.loc[tweets['prep_body'].isin(bot_tweets), 'writer'].unique()).astype(np.int8)

  bots = bot_check.loc[bot_check.sum(axis=1) > 1].index
  tweets['group'] = 'user'
  tweets.loc[tweets.writer.isin(bots), 'group'] = 'bot'

  # delete all bots
  indexWriters = tweets[tweets['group'] == 'bot'].index
  tweets.drop(indexWriters, inplace=True)
  tweets.drop(['group'], inplace=True, axis=1)
  print(f"Dropped all bot tweets: now {len(tweets)}")

  # drop except COL_NEEDED
  tweets.drop(tweets.columns.difference(COL_NEEDED), 1, inplace=True)

  if (len(tweets) > MAX_NUM_TWEETS):
    tweets = tweets.sample(frac=0.3).reset_index(drop=True)
    print(f"Too many tweets, sampled 30%: now {len(tweets)}")

  # save 'sentiment' as 'label'
  def map_sentiment_label(sentiment):
    if (sentiment == 'positive'):
      return 2
    elif (sentiment == 'neutral'):
      return 1
    return 0

  if 'sentiment' in COL_NEEDED:
    tweets['label'] = tweets['sentiment'].apply(map_sentiment_label)
    tweets = tweets.drop(['sentiment'], axis=1)

  return tweets


# remove overlap from test data for regression 
def remove_overlap(raw_test_data, train_data):
  print("removing overlapping data")
  not_overlaps = raw_test_data['body'].map(lambda body: body not in train_data['body'].unique())
  return raw_test_data[not_overlaps]

def split(tweets, train_ratio=0.5, eval_ratio=0.2, test_ratio=0.3):
  print("spliting sentiment labeled financial tweets")
  total_num = len(tweets)
  train_num = int(total_num * train_ratio)
  eval_num = int(total_num * eval_ratio)
  tweets = tweets.sample(frac=1).reset_index(drop=True)
  train = tweets.iloc[:train_num, :]
  eval = tweets.iloc[train_num:train_num+eval_num, :]
  test = tweets.iloc[train_num+eval_num:, :]

  print(f"train: {len(train)} / eval: {len(eval)} / test: {len(test)}")
  return train, eval, test

def main():
  cleaned_sentiment_labeled1 = clean(sentiment_labeled1)
  cleaned_sentiment_labeled2 = clean(sentiment_labeled2)
  sentiment_classification_data = pd.concat([cleaned_sentiment_labeled1, cleaned_sentiment_labeled2], ignore_index=True)
  sentiment_classification_train, sentiment_classification_eval, sentiment_classification_test = split(sentiment_classification_data)
  save(sentiment_classification_train, "sentiment-classification-train")
  save(sentiment_classification_test, "sentiment-classification-test")
  save(sentiment_classification_eval, "sentiment-classification-eval")

  cleaned_nasdaq_tweets = clean(nasdaq_tweets) 
  regression_data = remove_overlap(cleaned_nasdaq_tweets, sentiment_classification_data)
  save(regression_data, "regression-tweets-raw")

print("preprocessing started")
main()