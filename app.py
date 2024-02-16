from transformers import AutoTokenizer, DistilBertForSequenceClassification, RobertaForSequenceClassification
import torch
import gradio as gr
from torch import cuda
from scipy.special import softmax


# Requirements
def load_distilbert():
    model_path = "./results/sentiment_distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer

def load_roberta():
    model_path = "./results/sentiment_roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = RobertaForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer

def sentiment_analysis(model_type,text):

    if model_type == 'distilbert':
            model, tokenizer  = load_distilbert()
    else:
        model, tokenizer = load_roberta()

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        logits = model(**inputs).logits
    # predicted_class_id = logits.argmax().item()

    # scores_ = logits[0][0].detach
    scores_ = softmax(logits)


    return {'Negative':scores_[0][0], 'Positive':scores_[0][1]}

    # return model.config.id2label[predicted_class_id]

title = "Sentiment Analysis Application"
description = "This application assesses if a text is positive or negative"

model_type = gr.Radio(choices=['distilbert', 'roberta'], label='Select model type', value='distilbert' ) 

demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=[model_type,gr.TextArea(placeholder="Write your text here...")],
    outputs=["label"],
    examples=[["distilbert", "This is actually awesome :)"],["roberta", "I'm happy"]],
    title=title,
    description=description
)

demo.launch(share=False, server_name="0.0.0.0")