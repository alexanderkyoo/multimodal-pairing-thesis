import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

art_df = pd.read_csv('emotion_files/WikiArt_mapped.csv')
art_emotions = art_df[emotion_columns].fillna(0)
poem_df = pd.read_csv('initial_dataset/EmotionPoetryData-indexed.csv')
poem_emotions = poem_df[emotion_columns].fillna(0)
# art_meta = pd.read_csv('initial_dataset/WikiArt-info.tsv', sep='\t')

cosine_sim_matrix = cosine_similarity(art_emotions, poem_emotions)

art_ids = art_df['ID']
poem_ids = poem_df['ID']

top_pairings = []
for i, mapped_id in enumerate(art_ids):
    similarity_scores = cosine_sim_matrix[i]
    top_index = similarity_scores.argsort()[-2]
    top_poem = poem_ids.iloc[top_index]
    top_score = similarity_scores[top_index]
    top_pairings.append({'Art ID': mapped_id, 'Top Poem ID': top_poem, 'Sim Score': top_score})

top_pairings_df = pd.DataFrame(top_pairings)
top_pairings_df.to_csv('emotion_files/emotional_pairings.csv', index=False)

# merged_df = pd.merge(top_pairings_df, art_meta[['ID', 'Image URL']], left_on='Art ID', right_on='ID', how='left')
# merged_df = merged_df.drop(columns=['ID'])

# # merged_df = merged_df.sort_values(by='Similarity_Score', ascending=False)
# merged_df.to_csv('emotion_files/emotional_pairings.csv', index=False)

