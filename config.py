# config.py

import logging

from PIL import Image

JPG_DIR = "jpg"
PDF_DIR = "pdf"
PNG_DIR = "png"
CROPS_DIR = "crops"
CSV_DIR = "csv"
OCR_DIR = "ocr"

DPI = 450
BATCH_SIZE = 15
MAX_WORKERS = 2

MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

LOG_LEVEL = logging.INFO

VOTER_END_MARKER = Image.open("voter_end.jpg").convert("RGB")
