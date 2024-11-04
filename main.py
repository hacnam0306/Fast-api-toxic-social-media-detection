from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from ultralytics import YOLO
import cv2
import numpy as np
import requests
from io import BytesIO
import opennsfw2 as n2
from PIL import Image
import re

import nltk
from tensorflow.keras.models import load_model
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




app = FastAPI()

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK stopwords: {e}")


class Config:
    MODEL_PATH = "./model/best.pt"
    MAX_SEQUENCE_LENGTH = 400
    MAX_NUM_WORDS = 50000
    TEXT_MODEL_PATH = './model/LSTM_toxic_prediction_model.h5'
    CONF_CONTENT_THRESHOLD = 0.7
    IOU_CONTENT_THRESHOLD = 0.7
    CORNER_MODEL_PATH = "./OCR/weights/corner.pt"
    CONTENT_MODEL_PATH = "./OCR/weights/content.pt"
    FACE_MODEL_PATH = "./OCR/weights/face.pt"

class SimpleTokenizer:
    def __init__(self, max_words=50000):
        self.max_words = max_words
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}

    def fit_on_texts(self, texts):
        words = set()
        for text in texts:
            for word in text.lower().split():
                words.add(word)
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Sort words by frequency and take top max_words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_words]

        # Create word indices
        for i, (word, _) in enumerate(sorted_words, 1):  # Start from 1
            self.word_index[word] = i
            self.index_word[i] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.lower().split():
                if word in self.word_index:
                    seq.append(self.word_index[word])
            sequences.append(seq)
        return sequences


def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            padded_seq = [0] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)


class ImageURL(BaseModel):
    url: HttpUrl = Field(..., description="URL of the image to analyze")


class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.text_model = None
        self.tokenizer = None
        self._load_models()

    def _load_models(self):
        try:
            # Load YOLO model if needed
            if os.path.exists(Config.MODEL_PATH):
                logger.info("Loading YOLO model...")
                self.yolo_model = YOLO(Config.MODEL_PATH)

            # Load text classification model if needed
            if os.path.exists(Config.TEXT_MODEL_PATH):
                logger.info("Loading text classification model...")
                self.text_model = load_model(Config.TEXT_MODEL_PATH)

            # Initialize simple tokenizer
            logger.info("Initializing tokenizer...")
            self.tokenizer = SimpleTokenizer(max_words=Config.MAX_NUM_WORDS)

            # Initialize with basic toxic words vocabulary
            basic_vocab = [
                "this is a sample text",
                "toxic content example",
                "normal conversation",
                "hate speech example",
                "friendly conversation",
            ]
            self.tokenizer.fit_on_texts(basic_vocab)

            logger.info("Model loading completed")

        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            raise RuntimeError(f"Failed to load required models: {e}")

    def predict_text(self, text: str):
        # Preprocess text
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        sequences = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequences, Config.MAX_SEQUENCE_LENGTH)

        # Make prediction
        predictions = self.text_model.predict(padded)
        return predictions[0]


@app.post("/predict-nsfw")
async def predict_nsfw(image_url_request: ImageURL):
    try:
        response = requests.get(str(image_url_request.url))
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        nsfw_probability = float(n2.predict_image(image))

        return {"nsfw_probability": nsfw_probability}
    except Exception as e:
        logger.error(f"Error in NSFW prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-objects")
async def detect_objects(image_url: ImageURL):
    try:
        if model_manager.yolo_model is None:
            raise HTTPException(status_code=500, detail="YOLO model not loaded")

        response = requests.get(str(image_url.url))
        response.raise_for_status()

        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        results = model_manager.yolo_model(image)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                cls = int(box.cls.item())
                label = result.names[cls]


                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

        return {"detections": detections}
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# Initialize the model manager
model_manager = ModelManager()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)