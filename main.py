"""Script for Fast API Endpoint."""
import base64
import io
import logging
import warnings
from fastapi import FastAPI, HTTPException
from PIL import Image
import numpy as np
from pydantic import BaseModel
from src.engine import DefaultEngine
from src.model import DefaultModel

# Suppress specific warnings if necessary
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths
DETECTOR_CFG = "configs/craft_config.yaml"
DETECTOR_MODEL = "models/text_detector/craft_mlt_25k.pth"
RECOGNIZER_CFG = "configs/star_config.yaml"
RECOGNIZER_MODEL = "models/text_recognizer/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

# Initialize model and engine
try:
    model = DefaultModel(DETECTOR_CFG, DETECTOR_MODEL, RECOGNIZER_CFG, RECOGNIZER_MODEL)
    engine = DefaultEngine(model)
    logger.info("Model and engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize model or engine: {e}")
    raise RuntimeError("Initialization failed. Please check the configuration and model paths.")

class Item(BaseModel):
    image: str

@app.get("/")
def read_root():
    """Root endpoint to verify the API is running."""
    return {"message": "API is running..."}

@app.post("/ocr/predict")
def predict(item: Item):
    """Endpoint to predict text from an image."""
    try:
        img_bytes = base64.b64decode(item.image.encode("utf-8"))
        image = Image.open(io.BytesIO(img_bytes))
        image = np.array(image)
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        engine.predict(image)
        result = engine.result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error.")

    return result
