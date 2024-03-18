import io

from fastapi import FastAPI, File, UploadFile
# from skimage.feature import hog
# from skimage.transform import rescale
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
import pandas as pd
import cv2
import uvicorn

model = joblib.load("cnn_model.pkl")
# pca = joblib.load("pca.pkl")
# scaler = joblib.load("sc.pkl")

app = FastAPI()


@app.post("/predict_item")
def predict_item(file: UploadFile = File(...)) -> dict[str, float]:
    class_dict = {"MEL": 0., "NV": 0., "BCC": 0., "AKIEC": 0., "BKL": 0., "DF": 0., "VASC": 0.}
    image_vectors = []
    content = file.file.read()
    image = Image.open(BytesIO(content))
    image = np.asarray(image.convert("RGB"))
    image = cv2.resize(image, None, fx=0.3, fy=0.3)
    image_vectors.append(image)

    image_vectors_array = np.array(image_vectors)
    prediction = model.predict(image_vectors_array)
    prediction = np.round(prediction, decimals=5)
    c = 0
    for key, value in class_dict.items():
        class_dict[key] = prediction[0][c]
        c += 1

    return class_dict

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
