from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import onnx
import onnxruntime
import numpy as np

class Item(BaseModel):
    array: List[float]

app = FastAPI()

@app.post("/lenet")
async def lenet(item: Item):
    array = item.array
    image = np.array(array, dtype=np.double)
    image = np.reshape(image, (1,1,32,32)).astype(np.float32)
    onnx_model = onnx.load('cnn.onnx')
    session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    output = session.run(None, {'in': image})
    predictions = output[0][0].tolist()
    return {"predictions": predictions}

app.mount("/", StaticFiles(directory="static",html = True), name="static")
