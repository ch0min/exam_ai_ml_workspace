import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv("../OLA-2/data/raw/heart_2020_cleaned_transformed.csv")

# Sampling 1% of the dataset
sampled_df = df.sample(frac=0.01, random_state=42)

# Apply StandardScaler for Clustering
scaler = StandardScaler()
X_sampled = scaler.fit_transform(sampled_df.drop('HeartDisease', axis=1))

# Elbow Method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(X_sampled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# DBSCAN Clustering
# You might need to experiment with these parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_sampled)

# Add the cluster information to the DataFrame
sampled_df['Cluster'] = clusters

# Visualization
# Replace 'feature1' and 'feature2' with names of actual features you're interested in
plt.scatter(sampled_df['GenHealth'], sampled_df['PhysicalHealth'], c=sampled_df['Cluster'])
plt.xlabel('GenHealth')
plt.ylabel('PhysicalHealth')
plt.title('DBSCAN Clustering')
plt.show()

# Explore the resulting clusters
# For example, you can see the mean values of features in each cluster:
print(sampled_df.groupby('Cluster').mean())
