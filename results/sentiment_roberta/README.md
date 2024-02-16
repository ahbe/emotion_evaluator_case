---
language: 
- en
thumbnail: 
tags:
- text-classification
license: mit
datasets:
- imdb
metrics:
---

# IMDB Sentiment Task: roberta-base 

## Model description

A simple base roBERTa model trained on the "imdb" dataset.

## Intended uses & limitations

#### How to use

##### Transformers

```python
# Load model and tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use pipeline
from transformers import pipeline

model_name = "aychang/roberta-base-imdb"

nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

results = nlp(["I didn't really like it because it was so terrible.", "I love how easy it is to watch and get good results."])
```

##### AdaptNLP

```python
from adaptnlp import EasySequenceClassifier

model_name = "aychang/roberta-base-imdb"
texts = ["I didn't really like it because it was so terrible.", "I love how easy it is to watch and get good results."]

classifer = EasySequenceClassifier
results = classifier.tag_text(text=texts, model_name_or_path=model_name, mini_batch_size=2)
```

#### Limitations and bias

This is minimal language model trained on a benchmark dataset.

## Training data

IMDB https://huggingface.co/datasets/imdb

## Training procedure

#### Hardware
One V100

#### Hyperparameters and Training Args
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./models',
    overwrite_output_dir=False,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="steps",
    logging_dir='./logs',
    fp16=False,
    eval_steps=800,
    save_steps=300000
)
```

## Eval results

```
{'epoch': 2.0,
 'eval_accuracy': 0.94668,
 'eval_f1': array([0.94603457, 0.94731017]),
 'eval_loss': 0.2578844428062439,
 'eval_precision': array([0.95762642, 0.93624502]),
 'eval_recall': array([0.93472, 0.95864]),
 'eval_runtime': 244.7522,
 'eval_samples_per_second': 102.144}
```
