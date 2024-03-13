import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_pickle('../data/training.pkl')

df['HeartDisease'].value_counts().plot(kind='bar')
plt.title('Distribution of Target Variable (HeartDisease)')
plt.show()




df.hist(bins=50, figsize=(20, 15))
#plt.show()

#for column in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']:
#    plt.figure(figsize=(10,4))
#    sns.countplot(x=column, hue="HeartDisease", data=df)
#    plt.title(f'Distribution of {column} by HeartDisease')
#    plt.xticks(rotation=45)
#    plt.show()

df['HeartDisease'] = df['HeartDisease'].map({'No': 0, 'Yes': 1})  # Example for encoding

print(df)

# Compute the correlation matrix
#correlation_matrix = df.corr()
race_counts = df['Race'].value_counts()

# Creating a bar chart
plt.figure(figsize=(10, 6))
race_counts.plot(kind='bar')
plt.title('Distribution of Races')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotates the race names for better readability
plt.show()

# Visualize the correlation matrix
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
#plt.show()