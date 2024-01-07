from fastapi import FastAPI
import io
from fastapi.exceptions import HTTPException


#image recognition
from ultralytics import YOLO
from pydantic import BaseModel
import base64
from PIL import Image

#text_generation
from llama_cpp import Llama

app = FastAPI()

class PhotoData(BaseModel):
    photo_base64: str

class QuestionData(BaseModel):
    question: str
    
@app.post("/")
async def root():
    return {"message": "Hello World"}


@app.post("/imageRecognition")
async def yolo_recognition(photo_data: PhotoData):
    try:
        base64_str = photo_data.photo_base64
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        
        #model = YOLO('yolov8x.pt')
        model = YOLO('../models/best.pt')
        results = model.predict(img, show=True, conf=0.4)
        
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
            
@app.post("/textGeneration")
async def text_generation(question_arg: QuestionData):
    model_path = "/Users/artur/Desktop/BachelorsThesis/green_ai_llama/llama/llama-2-7b-chat/ggml-model-f32_q4_0.bin" #ggml-model-f16_q4_1.bin, ggml-model-f32_q4_0.bin
    model = Llama(model_path = model_path,
                #n_ctx = 512,            # context window size
                #n_gpu_layers = 1,        # enable GPU
                n_threads = 8,
                use_mlock = False)        # enable memory lock so not swap


    prompt = f"""
    [INST]<<SYS>>
    You are a helpful, respectful and honest assistant. The answer has to be short and coherent.
    Always answer as helpfully as possible, 
    while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, 
    dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in 
    nature. If a question does not make any sense, or is not factually coherent, explain why instead of 
    answering something not correct. If you don't know the answer to a question, please don't share false information.

    {question_arg}
    [/INST]
    """

    response = model(prompt = prompt, temperature = 0.2)
    
    return response


