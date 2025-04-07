import pandas as pd
import ast
import math
from nltk.corpus import wordnet as wn

poems = pd.read_csv("object_files/poetry_synsets.csv")
paintings = pd.read_csv("object_files/detected_images.csv")

def parse_synsets(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []

poems["Synsets"] = poems["Synsets"].apply(parse_synsets)

def extract_labels(synset_list):
    labels = set()
    for tup in synset_list:
        if isinstance(tup, tuple) and len(tup) > 0:
            syn_name = tup[0]
            try:
                syn = wn.synset(syn_name)
                labels.update(syn.lemma_names())
            except:
                continue
    return labels

poems["LabelSet"] = poems["Synsets"].apply(extract_labels)

def parse_objects(s):
    objects = {}
    if pd.isna(s) or s.strip() == "":
        return objects
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            label_part, rest = part.split(" (")
            label = label_part.strip()
            count_str = rest.split(",")[0].split("=")[1]
            conf_str = rest.split("max_conf=")[1].rstrip(")")
            count = int(count_str)
            conf = float(conf_str)
            objects[label] = (count, conf)
        except:
            continue
    return objects

paintings["ObjectDict"] = paintings["Detected_Objects"].apply(parse_objects)

def compute_score(object_dict, synset_list):
    matching = {}
    score = 0.0
    for label, (count, conf) in object_dict.items():
        for syn in synset_list:
            if isinstance(syn, tuple) and len(syn) > 0:
                syn_name = syn[0]
                try:
                    syn_lemmas = wn.synset(syn_name).lemma_names()
                    if label in syn_lemmas:
                        if label not in matching:
                            matching[label] = (count, conf)
                            score += conf * math.log1p(count)
                        break
                except:
                    continue
    return score, matching

results = []
for painting_id, obj_dict in zip(paintings["Painting"], paintings["ObjectDict"]):
    best_poem = None
    best_score = 0.0
    best_match_labels = []

    for poem_id, synsets in zip(poems["ID"], poems["Synsets"]):
        score, matched = compute_score(obj_dict, synsets)
        if score > best_score:
            best_score = score
            best_poem = poem_id
            best_match_labels = list(matched.keys())

    results.append({
        "Painting": painting_id,
        "Best_Matching_Poem_ID": best_poem if best_poem is not None else "",
        "Score": round(best_score, 4),
        "Matching_Objects": ", ".join(sorted(best_match_labels)) if best_match_labels else ""
    })
    if len(results) % 100 == 0:
        print(str(len(results)) + ' paintings recorded so far')

pd.DataFrame(results).to_csv("object_files/scoring_results.csv", index=False)