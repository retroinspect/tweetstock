"""
Finetune Bertweet to predict market sentiment of financial tweets
"""
import numpy as np
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer
from fintweet_sentiment_classification_model import BertweetForSequenceClassification
from utils import tokenize_function
"""
FILE specification
- columns=['body', 'label']
- dtype=[str, int]
- 0 <= label < num_classes
    - negative: 0
    - neutral : 1
    - positive: 2
"""

TRAIN_FILE='./sentiment-classification-train.csv'
EVAL_FILE='./sentiment-classification-eval.csv'
TEST_FILE='./sentiment-classification-test.csv'

raw_datasets = load_dataset('csv', data_files={
  "train": TRAIN_FILE,
  "eval": EVAL_FILE,
  "test": TEST_FILE
})

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["eval"]
test_dataset = tokenized_datasets["test"]

model = BertweetForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
training_args = TrainingArguments("fintweet-sentiment-classifier", evaluation_strategy="epoch")
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
_, _, metrics = trainer.predict(test_dataset=test_dataset)
print(metrics)

# TODO test accuracy: 91% -> accuracy 개선하기
model.save_pretrained("./fintweet-sentiment-classifier")
