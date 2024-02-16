import uvicorn
from fastapi import FastAPI, Response,Depends,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from torch import cuda
import torch
from jwtAuth import JWTBearer, create_access_token
from transformers import AutoTokenizer, DistilBertForSequenceClassification, RobertaForSequenceClassification
from scipy.special import softmax




app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

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

class JWTToken(BaseModel):
    token: str

class Request(BaseModel):
    text : str
    model_type: str = "distilbert"

class SentimentResponse(BaseModel):
    sentiment : str
    score : float


# ******************** Api to get JWT Token ********************

@app.get("/api/authenticate", summary="Get JWT access token",response_model=JWTToken)
async def authorize():

    # data to be signed using token
    data = {
        'info': 'secret information',
        'from': 'aiengine'
    }

    token = create_access_token(data=data)
    return {'token': token}

# ******************** Api to call using image file ********************
@app.post('/api/sentiment', summary="Extract information from images",response_model=SentimentResponse,dependencies=[Depends(JWTBearer())])
def scrape(req:Request):
        
    if req.model_type == "distilbert":
        tokenizer, model = load_distilbert()
        
    elif req.model_type == "roberta":
        tokenizer, model = load_roberta()
        

    inputs = tokenizer(req.text, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()

    output = model.config.id2label[predicted_class_id]

    scores_ = softmax(logits)


    result = {output : str(scores_[0][predicted_class_id])}

        
    return Response(json.dumps(result),media_type='application/json')



if __name__ == '__main__':
    uvicorn.run("sentiment:app", host="0.0.0.0", port=2024)