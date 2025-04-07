import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counts_df = pd.read_csv('survey_results/category_counts_overall.csv')

category_names = [
    "No Good Fit", 
    "Perceptual Analysis", 
    "Emotional Response", 
    "Interpretive Meaning-Making",
    "Personal Connection", 
    "Cross-Modal Integration", 
    "Technical/Evaluative"
]

plt.figure(figsize=(10, 6))

plt.scatter(counts_df['Category'], counts_df['Overall_Painting'], color='red', label='Painting', s=100)
plt.scatter(counts_df['Category'], counts_df['Overall_Pairing'], color='blue', label='Pairing', s=100)

plt.plot(counts_df['Category'], counts_df['Overall_Painting'], 'r--', alpha=0.6)
plt.plot(counts_df['Category'], counts_df['Overall_Pairing'], 'b--', alpha=0.6)

plt.xlabel('Response Category')
plt.ylabel('Count')
plt.title('Comparison of Response Categories: Painting vs. Pairing')

plt.xticks(range(7), category_names, rotation=45, ha='right')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('survey_results/category_comparison_overall.png', dpi=300)