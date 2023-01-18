import io
import PIL.Image as Image
import torch
import torchvision
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime

print(torch.__version__)
print(torchvision.__version__)

EMOTION_LABELS = ['happy', 'scary', 'calm', 'tender', 'melancholy']
MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def get_emotion_name(predict):
    return EMOTION_LABELS[(predict.index(max(predict)))]


class ModelLogic:
    def __init__(self):
        pass

    def model_specific_logic(self, image_bytes):
        started_time = datetime.now()

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = processor(text=EMOTION_LABELS, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        predict = list(logits_per_image.softmax(dim=1)[0])
        tags = [get_emotion_name(predict)]

        ended_time = datetime.now()
        print(f'TIME:{ended_time - started_time}')

        return [{'name': tag} for tag in tags]
