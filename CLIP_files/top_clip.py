import os
import sys
import pickle
import torch
import pandas as pd

with open('CLIP_files/shared_embedding_space.pkl', "rb") as f:
    embedding = pickle.load(f)
    
poem_embeddings = embedding["poem_embeddings"]
painting_embeddings = embedding["painting_embeddings"]
painting_names = embedding["painting_names"]
paintings = pd.read_csv('initial_dataset/WikiArt-info-truncated.tsv', sep='\t')

painting_id = sys.argv[1]
painting_row = paintings[paintings['ID'] == painting_id]
painting_index = painting_row.index[0]

similarity_scores = embedding["similarity_scores"]

painting_scores = similarity_scores[painting_index]

top = torch.topk(painting_scores, 5)
top_poems = list(zip(top.indices.tolist(), top.values.tolist()))

bottom = torch.topk(-painting_scores, 5)
bottom_poems = list(zip(bottom.indices.tolist(), [-score for score in bottom.values.tolist()]))

print("Top 5 Poems:")
for poem_id, score in top_poems:
    print(f"Poem Index: {poem_id}, Similarity Score: {score:.4f}\n")

print("\nBottom 5 Poems:")
for poem_id, score in bottom_poems:
    print(f"Poem Index: {poem_id}, Similarity Score: {score:.4f}\n")