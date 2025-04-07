import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

paintings = pd.read_csv('emotion_files/WikiArt_mapped.csv')
poems = pd.read_csv('initial_dataset/EmotionPoetryData-indexed.csv')

emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

painting_emotions = paintings[emotion_columns].fillna(0)
poem_emotions = poems[emotion_columns].fillna(0)

painting_id = sys.argv[1]
painting_row = paintings[paintings['ID'] == painting_id]
painting_index = painting_row.index[0]

sim = cosine_similarity([painting_emotions.iloc[painting_index]], poem_emotions)[0]

top = sim.argsort()[::-1][:5]
bottom = sim.argsort()[:5]

print("Top 5 Poem Matches:")
for idx in top:
    print(f"Poem: {poems['ID'].iloc[idx]} | Score: {sim[idx]:.4f}\n")

print("Bottom 5 Poem Matches:")
for idx in bottom:
    print(f"Poem: {poems['ID'].iloc[idx]} | Score: {sim[idx]:.4f}\n")