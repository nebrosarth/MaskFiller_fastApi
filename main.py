from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
from transformers.pipelines import PipelineException


class Item(BaseModel):
    text: str


app = FastAPI()
unmasker = pipeline("fill-mask", model="bert-base-uncased")


@app.get("/")
async def root():
    return {"message": "Welcome to Masked Language Model API"}


@app.post("/predict")
async def predict(item: Item):
    try:
        return unmasker.predict(item.text)
    except PipelineException as e:
        raise HTTPException(status_code=422, detail=str(e))
