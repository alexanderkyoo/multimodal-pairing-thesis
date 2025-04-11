import pandas as pd
import random

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
    df = pd.read_csv("survey_results_matching_cleaned.csv")
    response_columns = df.columns[1::2]  
    response_df = pd.DataFrame()

    for col in response_columns:
        score_col = df.columns[df.columns.get_loc(col) - 1]
        temp_df = pd.DataFrame({
            "Score_Column": score_col,
            "Response_Column": col,
            "Score": df[score_col],
            "Response": df[col]
        })
        response_df = pd.concat([response_df, temp_df], ignore_index=True)

    response_df["Category"] = ""
    return response_df

def categorize_responses():
    df = load_data()
    uncategorized = df[df["Category"] == ""].copy()
    uncategorized = uncategorized.sample(frac=1).reset_index(drop=True) 

    print("Categories:")
    for num, desc in CATEGORIES.items():
        print(f"{num}: {desc}")

    for idx, row in uncategorized.iterrows():
        response = row["Response"]
        if pd.isna(response) or response.strip() == "":
            df.loc[row.name, "Category"] = "0"
            continue

        print(f"\nRESPONSE:\n\"{response.strip()}\"")
        cats = input("Enter category numbers (comma-separated): ")

        df.loc[row.name, "Category"] = cats

    df.to_csv("survey_results/part2_explanation_categories.csv", index=False)

if __name__ == "__main__":
    categorize_responses()