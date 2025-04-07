import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import torch
import pandas as pd
import pickle

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#poems
poems = pd.read_csv('initial_dataset/poetry_truncated.csv')
poems_list = poems['Poem'].tolist()
text_inputs = processor(text=poems, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    poem_embeddings = model.get_text_features(**text_inputs)
    poem_embeddings /= poem_embeddings.norm(dim=-1, keepdim=True)

#paintings
painting_embeddings = []
painting_names = []
for painting_file in os.listdir("downloaded_paintings"):
    if painting_file.endswith((".jpg", ".png", ".jpeg")):
        image = Image.open(os.path.join("downloaded_paintings", painting_file))
        image_input = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_embedding = model.get_image_features(**image_input)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            
            painting_embeddings.append(image_embedding)
            painting_names.append(painting_file)

painting_embeddings = torch.stack(painting_embeddings).squeeze()

similarity_scores = painting_embeddings @ poem_embeddings.T
best_matches = similarity_scores.argmax(dim=1)

shared_embedding_space = {
    "poem_embeddings": poem_embeddings,
    "poems": poems_list, 
    "painting_embeddings": painting_embeddings,
    "painting_names": painting_names,
    "similarity_scores": similarity_scores
}

with open("shared_embedding_space.pkl", "wb") as f:
    pickle.dump(shared_embedding_space, f)
