from collections import Counter
import nltk
import pandas as pd

def preprocess_poem(poem):
    tokens = nltk.tokenize.word_tokenize(poem)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

def get_synsets(poem):
    words = preprocess_poem(poem)
    pos_tags = nltk.pos_tag(words)
    synsets = []
    for word, pos in pos_tags:
        wn_pos = get_wordnet_pos(pos)
        if wn_pos:  
            word_synsets = nltk.corpus.wordnet.synsets(word, pos=wn_pos)
            if word_synsets:
                synsets.append(word_synsets[0])
    synset_counts = Counter(synsets)
    return [(str(synset.name()), count) for synset, count in synset_counts.items()]

poetry_df = pd.read_csv("initial_dataset/poetry_truncated.csv")
poetry_df['Synsets'] = poetry_df['Poem'].apply(get_synsets)
poetry_df[['ID', 'Synsets']].to_csv("object_files/poetry_synsets.csv", index=False)