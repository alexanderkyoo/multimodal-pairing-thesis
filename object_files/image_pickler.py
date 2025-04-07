import pandas as pd
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import torch
import os
import pickle

pairings_df = pd.read_csv("painting_poem_pairings.csv")
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

def extract_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.transformer_decoder_class_predictions.squeeze().tolist()
    return embeddings

if os.path.exists("object_files/painting_embeddings.pkl"):
    with open("object_files/painting_embeddings.pkl", "rb") as f:
        embeddings_dict = pickle.load(f)
else:
    embeddings_dict = {}

for painting in pairings_df["Painting"]:
    if painting in embeddings_dict:
        continue
    image_path = os.path.join("downloaded_paintings", painting)
    if os.path.exists(image_path):
        embeddings = extract_embeddings(image_path)
    else:
        embeddings = None
    embeddings_dict[painting] = embeddings 
    with open("object_files/painting_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)
