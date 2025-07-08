import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi import FastAPI , UploadFile , File , HTTPException
from fastapi.responses import JSONResponse , FileResponse
import tensorflow as tf
import pickle
import numpy as np
import librosa
from extract_feature import extraction
from tempfile import NamedTemporaryFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

model = tf.keras.models.load_model(r"C:\Users\Lenovo\OneDrive\Desktop\Voice Emotion Det. FAST API\voice_emotion_model.h5")

enc_path = r"C:\Users\Lenovo\OneDrive\Desktop\Voice Emotion Det. FAST API\label_encoder.pkl"
with open(enc_path , 'rb') as f:
    encoder = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static" , StaticFiles(directory="static") , name = "static")

@app.post("/predict")
async def predict(file : UploadFile = File(...)):
        if not file.filename.lower().endswith((".wav" , ".mp3")):
            raise HTTPException(status_code=400 , detail="Only mp3 And wav Formate Audio Is Supported")
        

        with NamedTemporaryFile(delete=False , suffix=(".mp3")) as tempfile:
            readfile = await file.read()
            tempfile.write(readfile)
            temp_path = tempfile.name

        extracted = extraction(temp_path)
        raw = extracted[np.newaxis,...,np.newaxis]

        prediction = model.predict(raw)
        prediction_index = np.argmax(prediction)
        prediction_enc = encoder.inverse_transform([prediction_index])[0]
        confidence = float(prediction[0][prediction_index] * 100)

        os.unlink(temp_path)

        return JSONResponse({
             "Emotion" : prediction_enc,
             "Confidence" : confidence,
             "Status" : "Success"
        })

@app.get("/")
async def main():
    return FileResponse("static/index.html")