import sys, math, pandas as pd, ast
from nltk.corpus import wordnet as wn

def parse_detected_objects(obj_str):
    if pd.isna(obj_str): return []
    objs = []
    for part in obj_str.split(";"):
        part = part.strip()
        if " (" in part and ")" in part:
            label = part.split(" (")[0].strip()
            inner = part.split(" (")[1].rstrip(")")
            tokens = inner.split(",")
            try:
                count_val = int(tokens[0].split("=")[1].strip())
            except:
                count_val = 0
            try:
                conf_val = float(tokens[1].split("=")[1].strip())
            except:
                conf_val = 0.0
            objs.append((label, count_val, conf_val))
        else:
            objs.append((part, None, None))
    return objs

def is_match(obj_name, syn_name):
    try:
        syn = wn.synset(syn_name)
    except:
        return False
    words = set()
    for r in syn.closure(lambda s: s.hyponyms()):
        words.update(r.lemma_names())
    words.update(syn.lemma_names())
    return obj_name in words

if len(sys.argv) < 2:
    print("Usage: python script.py <painting_id>")
    sys.exit(1)
painting_id = sys.argv[1]

ps_df = pd.read_csv("../poetry_synsets.csv")
pt_df = pd.read_csv("../dataset/poetry_truncated.csv")
pt_df['Word_Length'] = pt_df['Poem'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
id_to_length = dict(zip(pt_df['ID'], pt_df['Word_Length']))

# Merge poetry synsets with truncated poems on 'ID'
poetry_df = pd.merge(ps_df, pt_df[['ID', 'Poem']], on="ID", how="left")
poetry_df["Synsets"] = poetry_df["Synsets"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
img_df = pd.read_csv("../image_identification_results.csv")
img_df["Detected_Objects"] = img_df["Detected_Objects"].apply(parse_detected_objects)
painting_to_objects = dict(zip(img_df["Painting"], img_df["Detected_Objects"]))
painting_objs = painting_to_objects.get(painting_id, [])
if not painting_objs:
    print(f"No detected objects for painting id {painting_id}")
    sys.exit(1)

final_results = []
for _, row in poetry_df.iterrows():
    poem_id = row["ID"]
    poem_text = row.get("Poem", "")
    if not isinstance(poem_text, str):
        poem_text = str(poem_text) if pd.notna(poem_text) else ""
    word_length = id_to_length.get(poem_id, len(poem_text.split()))
    if word_length >= 350:
        continue
    title = row["Title"]
    synsets = row["Synsets"]
    matching = {}
    score = 0.0
    for obj in painting_objs:
        label, count, conf = obj
        if count is None or conf is None: 
            continue
        for syn in synsets:
            if is_match(label, syn[0]):
                if label not in matching:
                    matching[label] = (count, conf)
                    score += conf * math.log1p(count)
                break
    final_results.append({
        "Painting": painting_id,
        "Poem_ID": poem_id,
        "Title": title,
        "Score": score,
        "Matching_Objects": ", ".join(matching.keys()),
        "Poem": poem_text,
        "Word_Length": word_length
    })

results_df = pd.DataFrame(final_results).sort_values("Score", ascending=False)
top_results = results_df.head(5)
bottom_results = results_df.tail(5)

print("Top 5 Poem Matches:", painting_id)
for _, row in top_results.iterrows():
    print(f"Poem: {row['Poem']}")
    print(f"Score: {row['Score']}")
    print(f"Matching Objects: {row['Matching_Objects']}\n")

print("Bottom 5 Poem Matches:", painting_id)
for _, row in bottom_results.iterrows():
    print(f"Poem: {row['Poem']}")
    print(f"Score: {row['Score']}")
    print(f"Matching Objects: {row['Matching_Objects']}\n")
