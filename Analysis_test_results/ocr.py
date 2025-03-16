# https://github.com/VikParuchuri/surya
# https://github.com/getomni-ai/zerox
# https://github.com/tesseract-ocr/tesseract
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

image = Image.open(IMAGE_PATH)
langs = ["en"] # Replace with your languages or pass None (recommended to use None)
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

predictions = recognition_predictor([image], [langs], detection_predictor)