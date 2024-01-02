from fastapi import FastAPI, Query
from ultralytics import YOLO
from pydantic import BaseModel
import base64
import io
from PIL import Image
from typing import Dict
from PIL import Image
from fastapi.exceptions import HTTPException

#text_generation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class PhotoData(BaseModel):
    photo_base64: str

class QuestionData(BaseModel):
    question: str
    
@app.post("/")
async def root():
    return {"message": "Hello World"}


@app.post("/recognition")
async def yolo_recognition(photo_data: PhotoData):
    try:
        base64_str = photo_data.photo_base64
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        
        #model = YOLO('yolov8x.pt')
        model = YOLO('../models/best.pt')
        results = model.predict(img, show=True)
        
        result_dict_list = []

        for box in results[0].boxes:
            class_id = results[0].names[box.cls[0].item()]
            result_dict = {"recognized_elemenets": class_id, 
                            "probability": box.conf[0].tolist(), 
                            "coordinates":box.xyxy[0].tolist()}
            
            result_dict_list.append(result_dict)

        # API returns only one detected object with the highest probability
        result_dict_highest_conf = result_dict_list[0]

        for result in result_dict_list[1:]:
            if result["probability"] > result_dict_highest_conf["probability"]:
                result_dict_highest_conf = result

        response = result_dict_highest_conf
        return response
    
    except Exception as e:
        # Handle exceptions if any
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
            
# @app.post("/text_generation")
# async def text_generation(question_arg: QuestionData):
#     # torch.set_default_device("cuda")
#     # torch.cuda.empty_cache()
    
#     question = question_arg.question
#     model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
#     inputs = tokenizer(question, return_tensors="pt", return_attention_mask=False)
#     outputs = model.generate(**inputs, max_length=200)
#     text = tokenizer.batch_decode(outputs)[0]
#     print(text)
