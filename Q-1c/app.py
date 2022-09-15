import pandas as pd

import mlflow
from sklearn.preprocessing import normalize

import uvicorn
from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel 

mlflow.set_tracking_uri("http://localhost:5000")
model_name      = "keras-ann-reg"
stage           = "Production"
model           = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Item(BaseModel):
    data: Union[List[Iris], None] = None

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")

@app.post("/predict")
async def predict(item: Item):
    dataInput           = item.dict()['data']
    data                = pd.DataFrame(dataInput)
    data[data.columns]  = normalize(data, axis=0)
    prediction          = model.predict(data).idxmax(axis=1).tolist()
    return {"Request":dataInput, "Response":prediction}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
