from enum import Enum
from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import time
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
from redis import asyncio as aioredis
from config import REDIS_HOST, REDIS_PORT
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from aioredis.exceptions import ResponseError
import uvicorn

app = FastAPI(title = "MediScan app")

modelMEL = joblib.load("LogRegForMEL.pkl")

def calc_square_means(img, val) :
    bot, top = 225-3*val, 225+3*val
    left, right = 300-4*val, 300+4*val
    sq = img[bot:top, left:right]
    return np.r_[sq.mean(axis=(0,1)), sq.std(axis=(0,1))]
def image_to_features(img):
    means = img.mean(axis=(0, 1))
    stds = img.std(axis=(0, 1))
    sq_res = calc_square_means(img, 40)
    return np.array([means[0], stds[0], sq_res[0], sq_res[3],
              means[1], stds[1], sq_res[1], sq_res[4],
              means[2], stds[2] ]).reshape(1,-1)

@app.get("/ping")
async def ping() -> str:
    return "Service is working"

@app.post("/predict")
async def predict_item(file: UploadFile = File(...)) -> str:
    content = file.file.read()
    image = Image.open(BytesIO(content))
    img = np.asarray(image.convert("RGB"))
    prob = modelMEL.predict_proba(image_to_features(img))
    timestamp = round(time.time() * 1000)
    # Can add the name of uploaded file of some kind of hash
    await redis.execute_command("RPUSH", "history",
                          "%d %.3f"%(timestamp, prob[0,1]))
    return "Probability %.3f"%(prob[0,1])

@app.post("/rate")
async def predict_item(rating: int) -> str:
    if rating in {1,2,3,4,5}:
        await redis.execute_command("RPUSH", "ratings", str(rating))
        return "Your rating is successfully saved."
    else :
        return "Wrong rating format: rating must be 1, 2, 3, 4 or 5."

@app.get("/current_rating")
async def current_rating() -> str:
    ratings = list(await redis.execute_command("LRANGE", "ratings", 0, -1))
    if len(ratings)==0 :
        return "No ratings recorded yet."
    ratings = [int.from_bytes(i) for i in ratings]
    rating = sum(ratings)/len(ratings)
    return f"Current rating is %.2f"%(rating)

@app.get('/history')
async def history() -> List[str]:
    history_log = list(await redis.execute_command("LRANGE", "history", 0, -1))
    # Maybe output in different format
    return history_log

@app.get('/')
def root():
    return dict()

@app.on_event("startup")
async def startup_event():
    redis = aioredis.from_url("redis://127.0.0.1:6380")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)