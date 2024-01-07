from fastapi import FastAPI, File, UploadFile
# from skimage.feature import hog
# from skimage.transform import rescale
import numpy as np
import joblib
# from PIL import Image
# from io import BytesIO
# import pandas as pd
import cv2

model = joblib.load("cnn_model.pkl")
# pca = joblib.load("pca.pkl")
# scaler = joblib.load("sc.pkl")

class_dict = {"MEL": 0., "NV": 0., "BCC": 0., "AKIEC": 0., "BKL": 0., "DF": 0., "VASC": 0.}

app = FastAPI()


@app.post("/predict_item")
async def predict_item(file: UploadFile = File(...)) -> dict:
    image_vectors = []
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, None, fx=0.3, fy=0.3)
    image_vectors.append(image)

    image_vectors_array = np.array(image_vectors)

    prediction = model.predict(image_vectors_array)
    c = 0
    for key, value in class_dict.items():
        class_dict[key] = prediction[0][c]
        c += 1

    return class_dict
