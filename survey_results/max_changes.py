import os
import pandas as pd

def rating_diff(paint_csv, pair_csv, pairing_type):
    rcols = ["angry","disgusted","fearful","joyful","sad","suprised"]
    df_paint = pd.read_csv(paint_csv)
    df_pair = pd.read_csv(pair_csv)
    df_paint["UserID"] = df_paint.iloc[:, 0]
    df_pair["UserID"] = df_pair.iloc[:, 0]
    merged = df_paint.merge(df_pair, on="UserID", suffixes=("_paint", "_pair"))
    
    painting_name = os.path.splitext(os.path.basename(paint_csv))[0]
    painting_number = painting_name.split("_")[0]
    
    rows = []
    for _, row in merged.iterrows():
        best_emotion = None
        best_diff = None
        for col in rcols:
            pval = row.get(f"{col}_paint")
            cval = row.get(f"{col}_pair")
            if pd.notna(pval) and pd.notna(cval):
                diff = cval - pval
                if best_diff is None or abs(diff) > abs(best_diff):
                    best_diff = diff
                    best_emotion = col
        
        if best_emotion is not None:
            explanation = row.get("Explanation_pair","")
            rows.append({
                "Painting Number": painting_number,
                "PairingType": pairing_type,
                "UserID": row["UserID"],
                "Emotion": best_emotion,
                "Difference": best_diff,
                "Explanation": explanation if pd.notna(explanation) else ""
            })
    
    return pd.DataFrame(rows)

def main():
    folder = "survey_results/pairing_questions"
    cats = ["CLIP","emo","obj","rand"]
    dfs = []
    
    for i in range(1,21):
        paint_csv = os.path.join(folder, f"{i}_painting.csv")
        if not os.path.exists(paint_csv):
            continue
        
        for c in cats:
            pair_csv = os.path.join(folder, f"{i}_pairing_{c}.csv")
            if os.path.exists(pair_csv):
                out = rating_diff(paint_csv, pair_csv, c)
                dfs.append(out)
    
    final = pd.concat(dfs, ignore_index=True)
    final.to_csv("survey_results/max_changes_all.csv", index=False)

if __name__ == "__main__":
    main()