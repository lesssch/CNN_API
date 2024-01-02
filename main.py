from fastapi import FastAPI, File, UploadFile
from skimage.feature import hog
from skimage.transform import rescale
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
import pandas as pd

model = joblib.load("sgd_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("sc.pkl")

class_dict = {"MEL": 1, "NV": 2, "BCC": 3, "AKIEC": 4, "BKL":5, "DF": 6, "VASC": 7}

app = FastAPI()
image_vectors = []

@app.post("/predict_item")
def predict_item(file: UploadFile = File(...)) -> str:
    content = file.file.read()
    image = Image.open(BytesIO(content))
    gray_image = np.asarray(image.convert("L"))

    image = rescale(gray_image, 1/3, mode='reflect')
    img_hog, hog_img = hog(
        image, pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys')
    flat_vector = np.array(hog_img).flatten()
    image_vectors.append(flat_vector)
    image_vectors_array = np.array(image_vectors)
    image_vectors_array = pca.transform(image_vectors_array)
    df = pd.DataFrame(data=image_vectors_array)
    df = scaler.transform(df)

    prediction = model.predict(df)
    result = None
    for key, value in class_dict.items():
        if value == prediction:
            result = key
            break

    return result
