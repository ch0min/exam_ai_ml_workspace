import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/heart_2020_cleaned.csv')

dict_replace = {'No':0, 'Yes' : 1}
df = df.replace(dict_replace)