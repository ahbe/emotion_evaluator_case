from transformers import AutoTokenizer, DistilBertForSequenceClassification, RobertaForSequenceClassification
from scipy.special import softmax
import json
from torch import cuda
import torch
import argparse
import pandas as pd
import string



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



parser = argparse.ArgumentParser()
parser.add_argument('--model_type', help='model name')
parser.add_argument('--input_csv', help='input csv file path')  
parser.add_argument('--output_csv', help='output csv file path')  


args = parser.parse_args()

# ******************** sentiment_analysis ********************
def sentiment_analysis(text, tokenizer,model):


    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()

    labels = ['negative', 'positive']

    output = labels[predicted_class_id]

    scores_ = softmax(logits)


    return output ,scores_[0][predicted_class_id]



# ******************** main ********************
def main(model_type,input_csv,output_csv):



    if model_type == 'distilbert':
        tokenizer, model = load_distilbert()

    elif model_type == 'roberta':
        tokenizer, model = load_roberta()

    try:
        df = pd.read_csv(input_csv)
    except Exception:
        df = pd.read_csv(input_csv, sep = ';',encoding='ISO-8859-1')

    sentiment = []
    score = []

    t = 0

    for i in df['review']:
        i = text_clean(i)
        try:
            temp_sentiment , temp_score =sentiment_analysis(i,tokenizer,model)
        except Exception as e:
            temp_sentiment , temp_score = None, None
            print(i)
            t+=1
        sentiment.append(temp_sentiment)
        score.append(temp_score)
    
    print("total fault",t)

    df_result = pd.DataFrame(columns=['sentiment', 'score'])

    df_result['sentiment'] = sentiment
    df_result['score'] = score

    df_result.to_csv(output_csv,index=False)

    


    


if __name__ == '__main__':

    main(args.model_type,args.input_csv,args.output_csv)