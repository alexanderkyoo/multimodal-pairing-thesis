import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

art_df = pd.read_csv('emotion_files/WikiArt_mapped.csv')
poem_df = pd.read_csv('initial_dataset/EmotionPoetryData-indexed.csv')
wikiart_info_df = pd.read_csv('initial_dataset//WikiArt-info.tsv', sep='\t')

emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

art_emotions = art_df[emotion_columns].fillna(0)
poem_emotions = poem_df[emotion_columns].fillna(0)

painting_id = sys.argv[1]
painting_row = art_df[art_df['ID'] == painting_id]
painting_index = painting_row.index[0]

cosine_sim = cosine_similarity([art_emotions.iloc[painting_index]], poem_emotions)[0]

top_indices = cosine_sim.argsort()[::-1][:5]
bottom_indices = cosine_sim.argsort()[:5]

wikiart_row = wikiart_info_df[wikiart_info_df['ID'] == painting_id]

print("Top 5 Poem Matches:")
for idx in top_indices:
    poem = poem_df['ID'].iloc[idx]
    score = cosine_sim[idx]
    print(f"  Poem: {poem}")
    print(f"  Similarity Score: {score:.4f}\n")

print("Bottom 5 Poem Matches:")
for idx in bottom_indices:
    poem = poem_df['ID'].iloc[idx]
    score = cosine_sim[idx]
    print(f"  Poem: {poem}")
    print(f"  Similarity Score: {score:.4f}\n")