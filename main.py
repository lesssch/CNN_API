from fastapi import FastAPI, File, UploadFile
from skimage.feature import hog
from skimage.transform import rescale
import numpy as np
import joblib
from PIL import Image
from io import BytesIO

model = joblib.load("sgd_model.pkl")
pca = joblib.load("pca.pkl")

class_dict = {"MEL": 1, "NV": 2, "BCC": 3, "AKIEC": 4, "BKL":5, "DF": 6, "VASC": 7}

app = FastAPI()

@app.post("/predict_item")
def predict_item(file: UploadFile = File(...)) -> str:
    content = file.file.read()
    image = Image.open(BytesIO(content))

    image = rescale(image, 1/3, mode='reflect')
    img_hog, hog_img = hog(
        image, pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys')
    flat_vector = np.array(hog_img).flatten()
    flat_vector = pca.transform(flat_vector)
    prediction = model.predict(flat_vector)
    result = None
    for key, value in class_dict.items():
        if value == prediction:
            result = key
            break

    return result
