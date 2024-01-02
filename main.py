import pickle
from fastapi import FastAPI, File, UploadFile
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
import numpy as np

pkl_filename = "./sgd_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

class_dict = {"MEL": 1, "NV": 2, "BCC": 3, "AKIEC": 4, "BKL":5, "DF": 6, "VASC": 7}

app = FastAPI()

@app.post("/predict_item")
def predict_item(file: UploadFile = File(...)) -> str:
    content = file.file.read()
    image = imread(content, as_gray=True)

    image = rescale(image, 1/3, mode='reflect')
    img_hog, hog_img = hog(
        image, pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys')
    flat_vector = np.array(hog_img).flatten()
    prediction = pickle_model.predict(flat_vector)
    result = None
    for key, value in class_dict.items():
        if value == prediction:
            result = key
            break

    return result
