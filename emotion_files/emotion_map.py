import pandas as pd

file_path = 'initial_dataset/WikiArt-Emotions-All.tsv'
df = pd.read_csv(file_path, sep='\t')

# Mappings
emotion_mapping = {
    'anger': ['ImageOnly: anger'],
    'disgust': ['ImageOnly: disgust'],
    'fear': ['ImageOnly: fear'],
    'joy': [
        'ImageOnly: happiness',
        'ImageOnly: love',
        'ImageOnly: gratitude',
        'ImageOnly: optimism'
    ],
    'neutral': ['ImageOnly: neutral'],
    'sadness': [
        'ImageOnly: sadness',
        'ImageOnly: regret',
        'ImageOnly: shame',
        'ImageOnly: shyness',
        'ImageOnly: pessimism'
    ],
    'surprise': ['ImageOnly: surprise']
}

mapped_emotions = pd.DataFrame(0.0, index=df.index, columns=emotion_mapping.keys())
for target_emotion, source_columns in emotion_mapping.items():
    for col in source_columns:
        if col in df.columns:
            mapped_emotions[target_emotion] += df[col]

mapped_emotions = mapped_emotions.div(mapped_emotions.sum(axis=1), axis=0).fillna(0)
metadata_columns = ['ID', 'Style', 'Category', 'Artist', 'Title']
df_metadata = df[metadata_columns]
df_result = pd.concat([df_metadata, mapped_emotions], axis=1)
output_file_path = 'emotion_files/WikiArt_mapped2.csv'
df_result.to_csv(output_file_path, index=False)