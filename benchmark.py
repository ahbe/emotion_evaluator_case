from transformers import AutoTokenizer, DistilBertForSequenceClassification, RobertaForSequenceClassification
from scipy.special import softmax
import json
from torch import cuda
import torch
import argparse
import pandas as pd
import string
from sklearn.metrics import accuracy_score


# ******************** text_clean ********************
def text_clean(text):
    punkt = list(string.punctuation)
    filtred_text = ''
    for x in text:
      if x not in punkt:
            filtred_text = filtred_text + x
    return filtred_text


# ******************** Loading distilbert ai model ********************

def load_distilbert():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./results/sentiment_distilbert")
        model = DistilBertForSequenceClassification.from_pretrained("./results/sentiment_distilbert")
    except Exception as e:
        raise e

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model


# ******************** Loading roberta ai model ********************

def load_roberta():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./results/sentiment_roberta")
        model = RobertaForSequenceClassification.from_pretrained("./results/sentiment_roberta")
    except Exception as e:
        raise e

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model


# ******************** sentiment_analysis ********************
def sentiment_analysis(text, tokenizer,model):


    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()

    labels = ['negative', 'positive']

    return labels[predicted_class_id]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path for data')

args = parser.parse_args()

# ******************** main ********************
def main(data_path):

    try:
        df = pd.read_csv(data_path)
    except Exception:
        df = pd.read_csv(data_path, sep = ';',encoding='ISO-8859-1')

    benchmark = {}

    for model_type in ["distilbert", "roberta"]:

        if model_type == 'distilbert':
            tokenizer, model = load_distilbert()

        elif model_type == 'roberta':
            tokenizer, model = load_roberta()



        sentiment = []

        for i in df['review']:
            i = text_clean(i)
            try:
                temp_sentiment =sentiment_analysis(i,tokenizer,model)
            except Exception as e:
                temp_sentiment = 'None'

            sentiment.append(temp_sentiment)
        
        benchmark[model_type] = accuracy_score(df['sentiment'],sentiment)



    print("benchmark results:")
    print(benchmark)
    


    


if __name__ == '__main__':

    main(args.data_path)