"""
- Given duration (start, end), calculate 
  1. stock price difference rate ((-1, inf))
  2. public sentiment ([-1, 1]) 
- Save as csv file
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from utils import save

# for real data generation
COMPANY_VALUES = "./data/raw/company-values.csv"
COMPANY_SENTIMENTS_DIR = "./data/company-sentiment"
PUBLIC_SENTIMENTS_DIR = "./data/regression"

# for debugging
# COMPANY_VALUES = "./data/sample/company-values.csv"
# COMPANY_SENTIMENTS_DIR = "./data/sample/company-sentiment"
# PUBLIC_SENTIMENTS_DIR = "./data/sample/regression"

companies = ['AAPL', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'MSFT']
# companies = ['AAPL', 'TSLA']
# companies = ['AAPL']

# start 부터 end까지, 끝점 포함 sentiment 평균 (left <= ser) & (ser <= right)
def public_sentiment(start, end, sentiment_data):
  sentiments = sentiment_data[sentiment_data['datetime'].between(start, end)]['sentiment_level']
  return sentiments.mean(), len(sentiments)

# start 직전의 종가에 대한 end 직후의 시가 변화율
def market_value_diff(start, end, company_value_data):
  try:
    last_day_before_start = company_value_data[company_value_data['day_date'] < start]['day_date'].idxmax()
    close_value_before_start = company_value_data['close_value'][last_day_before_start]
    first_day_after_end = company_value_data[company_value_data['day_date'] > end]['day_date'].idxmin()
    open_value_after_end = company_value_data['open_value'][first_day_after_end]
    return (open_value_after_end - close_value_before_start) / close_value_before_start
  except:
    return np.nan

def generate_date(days=3, start='2015-01-01', end='2019-12-31'):
  start_datetime = pd.to_datetime(start)
  end_datetime = pd.to_datetime(end)
  dates = []
  current_start = start_datetime
  current_end = start_datetime + pd.Timedelta(days, unit='D')
  while (current_end <= end_datetime):
    dates.append((current_start, current_end))
    current_start = current_end + pd.Timedelta(1, unit='D')
    current_end = current_start + pd.Timedelta(days, unit='D') 
  
  return dates

print("reading company values")
all_company_values = pd.read_csv(COMPANY_VALUES)
all_company_values.loc[:, 'day_date'] = all_company_values.loc[:, 'day_date'].apply(pd.to_datetime, errors='coerce') # change the type of object

print("generating dates")
dates = generate_date()

for ticker in companies:
  print(f"generating data for regression: {ticker}")
  sentiment_data = pd.read_csv(f"{COMPANY_SENTIMENTS_DIR}/nasdaq-tweet-sentiment-{ticker}.csv")
  sentiment_data.loc[:, 'datetime'] = sentiment_data.loc[:, 'datetime'].apply(pd.to_datetime, errors='coerce') # change the type of object
  company_value_data = all_company_values[all_company_values['ticker_symbol'] == ticker]
  row_list = []
  for (start, end) in dates:
    tweeter_sentiment, num_tweets = public_sentiment(start=start, end=end, sentiment_data=sentiment_data)
    market_diff = market_value_diff(start=start, end=end, company_value_data=company_value_data)
    # print(f"public sentiment: {tweeter_sentiment} market diff: {market_diff}")
    row_list.append({'tweeter_sentiment': tweeter_sentiment, 'num_tweets': num_tweets, 'market_diff': market_diff, 'start': start, 'end': end})

  df = pd.DataFrame(row_list, columns=('tweeter_sentiment', 'num_tweets', 'market_diff', 'start', 'end'))
  save(df, f"public-sentiment-{ticker}", PUBLIC_SENTIMENTS_DIR)

# TODO EDA 로 간격별 트윗 개수 큰 차이 없는지 보기
# MS의 경우 일주일에 80개 트윗이라 일주일을 간격으로하면 총 5*52=250개 포인트 -> 이정도면 괜찮은가?
# APPL의 경우 하루만 해도 엄청 많을듯 