import os
import pandas as pd

def extract_responses():
    folder = "survey_results/pairing_questions"
    pairing_types = {"CLIP": "CLIP", "emo": "emotion", "obj": "objective", "rand": "random"}
    rows = []
    
    for painting_number in range(1, 21):  # Paintings 1-20
        paint_csv = os.path.join(folder, f"{painting_number}_painting.csv")
        
        if not os.path.exists(paint_csv):
            continue
            
        df_paint = pd.read_csv(paint_csv)
        df_paint["UserID"] = df_paint.iloc[:, 0]
        paint_responses = {}
        for _, row in df_paint.iterrows():
            user_id = row["UserID"]
            paint_explanation = row.get("Explanation", "")
            
            if pd.isna(paint_explanation):
                paint_explanation = ""
            else:
                paint_explanation = paint_explanation.strip()
                
            paint_responses[user_id] = paint_explanation
        for code, full_name in pairing_types.items():
            pair_csv = os.path.join(folder, f"{painting_number}_pairing_{code}.csv")
            
            if not os.path.exists(pair_csv):
                continue
                
            df_pair = pd.read_csv(pair_csv)
            df_pair["UserID"] = df_pair.iloc[:, 0]
    
            for _, row in df_pair.iterrows():
                user_id = row["UserID"]
                pair_explanation = row.get("Explanation", "")
                
                if pd.isna(pair_explanation):
                    pair_explanation = ""
                else:
                    pair_explanation = pair_explanation.strip()
                if pair_explanation.strip() != "":
                    paint_explanation = paint_responses.get(user_id, "")
                    
                    rows.append({
                        "Painting Number": painting_number,
                        "Pairing Type": full_name,
                        "Participant ID": user_id,
                        "Painting Response": paint_explanation,
                        "Pairing Response": pair_explanation
                    })
    
    return pd.DataFrame(rows)

def main():
    response_df = extract_responses()
    
    if not response_df.empty:
        response_df.to_csv("survey_results/response_analysis_data.csv", index=False)
        print(f"Created file with {len(response_df)} response pairs")
    else:
        print("No matching responses found")

if __name__ == "__main__":
    main()