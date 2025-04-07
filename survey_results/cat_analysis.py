import pandas as pd

def count_categories(category_column):
    category_counts = {i: 0 for i in range(7)}
    
    for cell in category_column:
        if pd.notna(cell):
            categories = cell.split()
            for cat in categories:
                if cat.isdigit():
                    category_counts[int(cat)] += 1
    
    return category_counts

df = pd.read_csv('survey_results/categorized_responses.csv')

painting_counts = count_categories(df['Painting_Categories'])
pairing_counts = count_categories(df['Pairing_Categories'])

basis_types = df['Pairing Type'].unique()

all_counts = {
    'Category': list(range(7)),
    'Overall_Painting': [painting_counts[i] for i in range(7)],
    'Overall_Pairing': [pairing_counts[i] for i in range(7)]
}

for basis in basis_types:
    basis_df = df[df['Pairing Type'] == basis]
    
    basis_painting_counts = count_categories(basis_df['Painting_Categories'])
    basis_pairing_counts = count_categories(basis_df['Pairing_Categories'])
    
    all_counts[f'{basis}_Painting'] = [basis_painting_counts[i] for i in range(7)]
    all_counts[f'{basis}_Pairing'] = [basis_pairing_counts[i] for i in range(7)]
    
pd.DataFrame(all_counts).to_csv('survey_results/category_counts_overall.csv', index=False)