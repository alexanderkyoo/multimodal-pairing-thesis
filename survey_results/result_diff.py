import pandas as pd

def abs_diff():
    paintings = pd.read_csv("survey_results/survey_results_paintings.csv")
    changes = pd.read_csv("survey_results/max_changes_all.csv")
    changes["AbsDiff"] = changes["Difference"].abs()
    paintings["Abstract"] = paintings["Abstract"].str.strip().eq("True")
    paintings["Object"] = paintings["Object"].str.strip().eq("True")
    merged = changes.merge(paintings[["Painting Number", "Abstract", "Object"]],
                           on="Painting Number", how="left")
    types = ["CLIP", "emo", "obj", "rand"]
    cols = ["All", "abstract", "non_abstract", "object", "non_object"]
    out = pd.DataFrame(index=types, columns=cols)
    
    for t in types:
        sub = merged[merged["PairingType"] == t]
        out.loc[t, "All"] = sub["AbsDiff"].mean()
        a = sub[sub["Abstract"] == True]
        out.loc[t, "abstract"] = a["AbsDiff"].mean()
        na = sub[sub["Abstract"] == False]
        out.loc[t, "non_abstract"] = na["AbsDiff"].mean()
        o = sub[sub["Object"] == True]
        out.loc[t, "object"] = o["AbsDiff"].mean()
        no = sub[sub["Object"] == False]
        out.loc[t, "non_object"] = no["AbsDiff"].mean()

    return out

def rel_diff():
    paintings = pd.read_csv("survey_results/survey_results_paintings.csv")
    changes = pd.read_csv("survey_results/max_changes_all.csv")
    changes["AbsDiff"] = changes["Difference"]
    paintings["Abstract"] = paintings["Abstract"].str.strip().eq("True")
    paintings["Object"] = paintings["Object"].str.strip().eq("True")
    merged = changes.merge(paintings[["Painting Number", "Abstract", "Object"]],
                           on="Painting Number", how="left")
    types = ["CLIP", "emo", "obj", "rand"]
    cols = ["All", "abstract", "non_abstract", "object", "non_object"]
    out = pd.DataFrame(index=types, columns=cols)
    
    for t in types:
        sub = merged[merged["PairingType"] == t]
        out.loc[t, "All"] = sub["AbsDiff"].mean()
        a = sub[sub["Abstract"] == True]
        out.loc[t, "abstract"] = a["AbsDiff"].mean()
        na = sub[sub["Abstract"] == False]
        out.loc[t, "non_abstract"] = na["AbsDiff"].mean()
        o = sub[sub["Object"] == True]
        out.loc[t, "object"] = o["AbsDiff"].mean()
        no = sub[sub["Object"] == False]
        out.loc[t, "non_object"] = no["AbsDiff"].mean()

    return out

def main():
    
    out1 = abs_diff()
    out1 = out1.astype(float).round(3)   
    out1.to_csv("survey_results/mean_abs_diff.csv", index_label="Basis")

    out2 = rel_diff()
    out2 = out2.astype(float).round(3)   
    out2.to_csv("survey_results/mean_rel_diff.csv", index_label="Basis")


if __name__ == "__main__":
    main()
