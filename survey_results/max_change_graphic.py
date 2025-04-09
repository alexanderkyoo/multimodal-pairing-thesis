import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("survey_results/mean_rel_diff.csv")
categories = ["All", "abstract", "non_abstract", "object", "non_object"]
n_categories = len(categories)

bases = df['Basis'].tolist()

data = {}
for i, basis in enumerate(bases):
    data[basis] = df.iloc[i, 1:].values.astype(float)


x = np.arange(n_categories)
offsets = [(-0.3) + j * 0.2 for j in range(len(bases))]

global_mean = 0.230 # grabbed directly as full survey is omitted
fig, ax = plt.subplots(figsize=(12, 7))

for basis, offset in zip(bases, offsets):
    ax.bar(x + offset, data[basis], 0.2, label=basis)

ax.axhline(global_mean, color='black', linestyle='--', linewidth=2, label=f"Global Mean ({global_mean})")

ax.set_xlabel("Category")
ax.set_ylabel("Mean Value")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("survey_results/part_1_quant.png")
