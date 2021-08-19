# tweetstock: Regression Analysis from Public Tweet Sentiment toward NASDAQ stock prices
Given tweets about NASDAQ top 6 stocks(AAPL, GOOG, GOOGL, TSLA, AMZN, MSFT), will be there any relationship between tweet sentiments and the stock price?

## Directory structure
|- data
  |- company-sentiment
    : sentiment labeled tweet ids on NASDAQ stocks
  |- raw
    |- company-tweets.csv
    |- company-values.csv
    |- nasdaq-tweets.csv
    |- (sentiment labeled financial tweets)
  |- regression
    : data for regression of public tweet sentiment and market value
  |- sample
    : data for debugging data generator
  |- sentiment-classifier
    : data to finetune BERtweet for sentiment classification of financial tweets
|- calculate.py
  : calculate public sentiment and stock price difference of nasdaq stocks for intervals 
|- finetune_classifier.py
  : finetune model from `fintweet_sentiment_classifier.py`
|- fintweet_sentiment_classifier.py
  : model copied & slightly modified from HuggingFace Transformer RoBERTa
|- nasdaq_tweet_sentiment_tagger.py
  : tag sentiment to nasdaq tweets with fine-tuned classifier
|- preproces_tweets.py
  : drop duplicated and suspicious nasdaq tweets and sentiment labeled financial tweets
|- regression.py
  : output the relation between public sentiment and stock price direction
|- utils.py
  : misc functions

## Experiement Pipeline
### Step 1. Preprocess tweets to get rid of spam
- invovles `preproces_tweets.py`
- Drop duplicated or suspicious spam tweets
- Spam filter was based on \[Kaggle notebook of aramacus] \[1]
- Due to limitation of computing power, sampled 30% of the tweets
### Step 2. Finetune BERtweet with sentiment labeled financial tweets
- invovles `fintweet_sentiment_classifier.py`, `finetune_classifier.py`
- To make sentiment classifier for financial tweets, finetune BERtweet (which is RoBERTa pretrained on tweeter data) with sentiment labeled financial tweets
### Step 3. Caculate public sentiment & stock price difference of NASDAQ tweets
- invovles `nasdaq_tweet_sentiment_tagger.py`, `calculate.py`
- Tag sentiment to NASDAQ tweets 
- Calculate public sentiment for 3 days
- Calculate price difference of: 
  - open price of before the duration
  - close price of after the duration
### Step 4. Regression 
- involves `regression.py`
- Output the relation between public sentiment and stock price direction
- Trained for tweets of 2015/01/01-2017/12/31
- Tested for tweets of 2018/01/01-2019/12/31

## Results
|       | Train (accuracy) | Test (accuracy) |
|-------|------------------|-----------------|
| AAPL  | 63.5%            | 74.2%           |
| GOOG  | 66.4%            | 60.9%           |
| GOOGL | 65.6%            | 65.2%           |
| AMZN  | 63.9%            | 62.6%           |
| TSLA  | 62.2%            | 54.9%           |
| MSFT  | 58.1%            | 66.4%           |

## Discussion

## References
\[1]: https://www.kaggle.com/aramacus/bot-hunting-or-how-many-tweets-were-made-by-bots
\[2]:
@inproceedings{bertweet,
title     = {{BERTweet: A pre-trained language model for English Tweets}},
author    = {Dat Quoc Nguyen and Thanh Vu and Anh Tuan Nguyen},
booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
year      = {2020},
pages     = {9--14}
}