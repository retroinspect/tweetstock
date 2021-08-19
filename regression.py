from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

companies = ['AAPL', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'MSFT']
DATA_DIR = "./data/regression"

# TODO feature engineering
for ticker in companies:
  print(f"regression: {ticker}")
  data = pd.read_csv(f"{DATA_DIR}/public-sentiment-{ticker}.csv")
  data.dropna(inplace=True)

  # drop if num of tweets is too small (< 50) 
  data = data[data['num_tweets'] > 50]

  train_data = data[pd.to_datetime(data['end']) < pd.to_datetime('2018-01-01')]
  test_data = data[pd.to_datetime(data['end']) >= pd.to_datetime('2018-01-01')]

  train_features = train_data[['tweeter_sentiment', 'num_tweets']]
  train_labels = train_data['market_diff'].map(lambda diff: diff >= 0)

  test_features = test_data[['tweeter_sentiment', 'num_tweets']]
  test_labels = test_data['market_diff'].map(lambda diff: diff >= 0)

  scaler = StandardScaler()
  train_features = scaler.fit_transform(train_features)
  test_features = scaler.transform(test_features)

  model = LogisticRegression()
  model.fit(train_features, train_labels)
  model.predict(test_features)

  train_accuracy = "{:.2f}".format(model.score(train_features, train_labels) * 100)
  test_accuracy = "{:.2f}".format(model.score(test_features, test_labels) * 100)
  print(f"train accuracy: {train_accuracy}% \ntest accuracy: {test_accuracy}%")

  coeff_tweeter_sentiment = "{:.2f}".format(model.coef_[0][0])
  coeff_num_tweets = "{:.2f}".format(model.coef_[0][1])
  # print(f"tweeter_sentiment: {coeff_tweeter_sentiment} / num_tweets: {coeff_num_tweets}")