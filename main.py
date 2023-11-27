from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
unmasker = pipeline("fill-mask", model="bert-base-uncased")


@app.get("/")
async def root():
    return {"message": "Welcome to Masked Language Model API"}


@app.post("/predict")
async def predict(item: Item):
    return unmasker.predict(item.text)
