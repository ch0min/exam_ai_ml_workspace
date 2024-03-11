import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/heart_2020_cleaned.csv')

dict_replace = {'No': 0, 'Yes': 1}
df = df.replace(dict_replace)

dict_sex = {'Female': 0, 'Male': 1}
df = df.replace(dict_replace)


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight (< 18.5)'
    elif 18.5 <= bmi < 25.0:
        return 'Normal weight (18.5 - 25.0)'
    elif 25.0 <= bmi < 30.0:
        return 'Overweight (25.0 - 30.0)'
    else:
        return 'Obese (30 <)'

# Apply the function to the DataFrame
df['BMI'] = df['BMI'].apply(categorize_bmi)

dict_BMI = {'Underweight (< 18.5)': 0, 'Normal weight (18.5 - 25.0)': 1,
            'Overweight (25.0 - 30.0)': 2, 'Obese (30 <)': 3}
df = df.replace(dict_BMI)

age_ranges = df["AgeCategory"].unique()

age_codes, _ = pd.factorize(age_ranges, sort=True)

age_range_to_code = dict(zip(age_ranges, age_codes))

df["AgeCategory"] = df["AgeCategory"].map(age_range_to_code)


print(df["AgeCategory"])










#df_training.to_pickle('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/training.pkl')
