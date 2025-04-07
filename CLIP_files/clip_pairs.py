import os
import torch
import pickle
import pandas as pd

with open("CLIP_files/shared_embedding_space.pkl", "rb") as f:
    embedding = pickle.load(f)

poem_embeddings = embedding["poem_embeddings"]
painting_embeddings = embedding["painting_embeddings"]
painting_ids = embedding["painting_names"]
poems = embedding["poems"]
similarity_scores = embedding["similarity_scores"]

best_matches = similarity_scores.argmax(dim=1)
pairings = []

for i, painting in enumerate(painting_ids):
    best_poem_index = best_matches[i].item() 
    similarity_score = similarity_scores[i, best_poem_index].item()
    pairings.append({"Painting": painting, "Best Matching Poem Index": best_poem_index, "Similarity Score": similarity_score})

pd.DataFrame(pairings).sort_values(by="Similarity Score", ascending=False).to_csv("CLIP_files/clip_pairings.csv", index=False)
