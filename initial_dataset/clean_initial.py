import pandas as pd

# Truncate Poetry Foundation data
poetry_df = pd.read_csv('initial_dataset/PoetryFoundationData.csv')
poetry_truncated_df = poetry_df.head(5000)
poetry_truncated_path = 'initial_dataset/poetry_truncated.csv'
poetry_truncated_df.to_csv(poetry_truncated_path, index=True, index_label='ID')

# Truncate WikiArt data
wikiart_df = pd.read_csv('initial_dataset/WikiArt-info.tsv', sep='\t')
wikiart_truncated_df = wikiart_df.head(2000)
wikiart_truncated_path = 'initial_dataset/WikiArt-info-truncated.tsv'
wikiart_truncated_df.to_csv(wikiart_truncated_path, index=False, sep='\t')

ep_df = pd.read_csv('initial_dataset/EmotionPoetryData.csv')
ep_df.to_csv('initial_dataset/EmotionPoetryData-indexed.csv', index=True, index_label='ID')