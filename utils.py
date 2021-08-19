from transformers import AutoTokenizer
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
def tokenize_function(examples):
  return tokenizer(examples["body"], padding="max_length", truncation=True)

def save(tweets, filename, directory="./data"):
  tweets.to_csv(f'{directory}/{filename}.csv', sep=',', na_rep='NaN', index=False)
  print(f"Saved into {filename}.csv")
