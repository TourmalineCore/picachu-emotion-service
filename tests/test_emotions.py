import json
import os
import glob
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score
from model.emotion_model import model
from model.emotion_model import processor
from model.emotion_model import EMOTION_LABELS


DEVICE = 'cpu'

os.system('unzip ./tests/test_photos.zip -d ./tests')

labels = os.listdir('./tests/test_photos/')

image_list = []
labels_list = []

for label in labels:
    for image_path in glob.glob(f'./tests/test_photos/{label}/*.jpg'):
        image = Image.open(image_path)
        image_list.append(image)
        labels_list.append(label)

model.to(DEVICE)
model.eval()

y_preds = []

for image in image_list:
    inputs = processor(text=EMOTION_LABELS, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs.to(DEVICE))
    logits_per_image = outputs.logits_per_image
    predict = list(logits_per_image.softmax(dim=1)[0])
    y_preds.append(EMOTION_LABELS[(predict.index(max(predict)))])

acc = accuracy_score(labels_list, y_preds)
prec = precision_score(labels_list, y_preds, average='weighted')
rec = recall_score(labels_list, y_preds, average='weighted')

with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": acc, "precision": prec, "recall": rec}, outfile)
    