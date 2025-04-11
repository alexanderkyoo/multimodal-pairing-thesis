import pandas as pd
import re
import os
from collections import Counter, defaultdict

CATEGORIES = {
    1: "Perceptual Analysis",
    2: "Emotional Response",
    3: "Interpretive Meaning-Making",
    4: "Personal Connection",
    5: "Cross-Modal Integration",
    6: "Technical/Evaluative Comments",
    0: "Not a good fit"
}

df = pd.read_csv("survey_results/part2_explanation_categories.csv")

category_cols = [col for col in df.columns if col.endswith("_Categories")]

category_counts = {}
all_counts = Counter()

for col in category_cols:
    for val in df[col].dropna():
        categories = re.findall(r"\d+", str(val))
        for cat in categories:
            if col not in category_counts:
                category_counts[col] = defaultdict(int)
            category_counts[col][cat] += 1
            all_counts[cat] += 1

categories = sorted(set(cat for counts in category_counts.values() for cat in counts))
columns = ["All"] + category_cols

table_data = []
for cat in categories:
    row = [all_counts.get(cat, 0)]
    for col in category_cols:
        row.append(category_counts[col].get(cat, 0))
    table_data.append(row)

category_names = [CATEGORIES.get(int(cat), f"Unknown Category {cat}") for cat in categories]

table_df = pd.DataFrame(table_data, columns=columns, index=category_names)

table_df.to_csv('survey_results/p2_explanation.csv')