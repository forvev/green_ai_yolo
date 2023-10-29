from fastapi import FastAPI, Query
from ultralytics import YOLO
from pydantic import BaseModel
import base64
import io
from PIL import Image
from typing import Dict
from PIL import Image

app = FastAPI()

class PhotoData(BaseModel):
    photo_base64: str

@app.post("/")
async def root():
    return {"message": "Hello World"}


@app.post("/recognition")
async def yolo_recognition(photo_data: PhotoData):
    try:
        base64_str = photo_data.photo_base64
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        
        model = YOLO('yolov8n.pt')
        results = model(img, show=True)

        response = {"message": f"Received argument: {results}"}
        return response
    except Exception as e:
        # Handle exceptions if any
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
        return {"message": "message"}