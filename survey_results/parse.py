import pandas as pd
import json
import math
import os

def split(df, matching_columns):
    df_match = df[matching_columns]
    remaining_columns = [col for col in df.columns if (col not in matching_columns)]
    df_other = df[remaining_columns]
    return df_match, df_other

def clean(df, labels):
    for col in df.columns:
        df.loc[:, col] = df[col].apply(lambda x: labels[x] if x in labels else x)
    return df

def split_multi(df, output_prefix, file_names, columns):
    num_column = len(columns)  # how many columns you want per file
    total_columns = len(df.columns)
    num_splits = math.ceil(total_columns / num_column)
    
    for i in range(num_splits):
        start_col = i * num_column
        end_col = start_col + num_column
        subset_cols = df.columns[start_col:end_col]
        subset_df = df[subset_cols]

        new_col_names = columns
        subset_df.columns = new_col_names
        output_filename = output_prefix + file_names[i] + '.csv'
        subset_df.to_csv(output_filename, index=True)

    rating_cols = ["angry","disgusted","fearful","joyful","sad","suprised"]
    temp_df = df.copy()
    temp_df.dropna(how='all', subset=rating_cols, inplace=True)
    m = temp_df[rating_cols].mean()
    return tuple(m[col] for col in rating_cols)

def main():
    # split data into matching and pairing
    df = pd.read_csv('survey_results/survey_results_full_questions.csv')
    matching_columns = ['Q31', 'Q41', 'Q33', 'Q40', 'Q34', 'Q39', 'Q35', 'Q38']
    df_matching, df_pairing = split(df, matching_columns)
    df_matching = df_matching.iloc[2:]
    df_pairing = df_pairing.iloc[2:]
    df_matching.to_csv('survey_results/survey_results_matching_qs.csv', index=False)
    df_pairing.to_csv('survey_results/survey_results_painting_pairing.csv', index=False)

    # clean matching data to fit basis labels
    with open('survey_results/matching_labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    df_cleaned_mc = clean(df_matching, labels)
    df_cleaned_mc.to_csv('survey_results/survey_results_matching_cleaned.csv', index=False)

    # split pairing questions by pairing
    column_names = ['angry','disgusted','fearful','joyful','sad','suprised','Explanation']
    file_names = [
        '1_painting', '2_painting', '3_painting', '4_painting', '5_painting',
        '1_pairing_CLIP', '1_pairing_emo', '1_pairing_obj', '1_pairing_rand',
        '2_pairing_CLIP', '2_pairing_emo', '2_pairing_obj', '2_pairing_rand',
        '3_pairing_CLIP', '3_pairing_emo', '3_pairing_obj', '3_pairing_rand',
        '4_pairing_CLIP', '4_pairing_emo', '4_pairing_obj', '4_pairing_rand',
        '5_pairing_CLIP', '5_pairing_emo', '5_pairing_obj', '5_pairing_rand',
        '6_painting', '7_painting', '8_painting', '9_painting', '10_painting',
        '6_pairing_CLIP', '6_pairing_emo', '6_pairing_obj', '6_pairing_rand',
        '7_pairing_CLIP', '7_pairing_emo', '7_pairing_obj', '7_pairing_rand',
        '8_pairing_CLIP', '8_pairing_emo', '8_pairing_obj', '8_pairing_rand',
        '9_pairing_CLIP', '9_pairing_emo', '9_pairing_obj', '9_pairing_rand',
        '10_pairing_CLIP', '10_pairing_emo', '10_pairing_obj', '10_pairing_rand',
        '11_painting', '12_painting', '13_painting', '14_painting', '15_painting',
        '11_pairing_CLIP', '11_pairing_emo', '11_pairing_obj', '11_pairing_rand',
        '12_pairing_CLIP', '12_pairing_emo', '12_pairing_obj', '12_pairing_rand',
        '13_pairing_CLIP', '13_pairing_emo', '13_pairing_obj', '13_pairing_rand',
        '14_pairing_CLIP', '14_pairing_emo', '14_pairing_obj', '14_pairing_rand',
        '15_pairing_CLIP', '15_pairing_emo', '15_pairing_obj', '15_pairing_rand',
        '16_painting', '17_painting', '18_painting', '19_painting', '20_painting',
        '16_pairing_CLIP', '16_pairing_emo', '16_pairing_obj', '16_pairing_rand',
        '17_pairing_CLIP', '17_pairing_emo', '17_pairing_obj', '17_pairing_rand',
        '18_pairing_CLIP', '18_pairing_emo', '18_pairing_obj', '18_pairing_rand',
        '19_pairing_CLIP', '19_pairing_emo', '19_pairing_obj', '19_pairing_rand',
        '20_pairing_CLIP', '20_pairing_emo', '20_pairing_obj', '20_pairing_rand'
        ]
    folder = "survey_results/pairing_questions/"
    split_multi(df_pairing, folder, file_names, column_names)
    

if __name__ ==  "__main__":
    main()