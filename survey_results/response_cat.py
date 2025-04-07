import os
import pandas as pd
import csv

CATEGORIES = {
   1: "Perceptual Analysis",
   2: "Emotional Response",
   3: "Interpretive Meaning-Making",
   4: "Personal Connection",
   5: "Cross-Modal Integration",
   6: "Technical/Evaluative Comments",
   0: "Not a good fit"
}

def load_data():
   df = pd.read_csv("survey_results/response_analysis_data.csv")
   cat_df = df.copy()
   cat_df["Painting_Categories"] = ""
   cat_df["Pairing_Categories"] = ""
   return df, cat_df

def categorize_responses():
   df, cat_df = load_data()
   print("Categories:")
   for num, desc in CATEGORIES.items():
       print(f"{num}: {desc}")
  
   uncategorized_items = []
   for idx, row in df.iterrows():
       painting_num = row["Painting Number"]
       pairing_type = row["Pairing Type"]
       participant_id = row["Participant ID"]
       painting_response = row["Painting Response"]
       pairing_response = row["Pairing Response"]
      
       uncategorized_items.append((idx, "painting", painting_num, pairing_type, participant_id, painting_response))
       uncategorized_items.append((idx, "pairing", painting_num, pairing_type, participant_id, pairing_response))
  
   for idx, response_type, painting_num, pairing_type, participant_id, response in uncategorized_items:
       if pd.isna(response) or response.strip() == "":
           if response_type == "painting":
               cat_df.loc[idx, "Painting_Categories"] = "0"
           else:
               cat_df.loc[idx, "Pairing_Categories"] = "0"
           continue
       
       print(f"RESPONSE: \"{response}\"")
       cats = input()
       
       if response_type == "painting":
           cat_df.loc[idx, "Painting_Categories"] = cats
       else:
           cat_df.loc[idx, "Pairing_Categories"] = cats
  
   cat_df.to_csv("survey_results/categorized_responses.csv", index=False)

if __name__ == "__main__":
   categorize_responses()