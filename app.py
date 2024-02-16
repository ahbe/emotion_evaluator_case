from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import gradio as gr
from scipy.special import softmax

#setup
model_path = "./results/sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

def sentiment_analysis(text):

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


demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.TextArea(placeholder="Write your text here..."),
    outputs=["label"],
    examples=[["I'm happy"],["I'm sad "]],
    title=title,
    description=description
)

demo.launch(share=False, server_name="0.0.0.0")