import pandas as pd
import numpy as np
<<<<<<< HEAD
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
=======
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Loading the dataset.
# df = pd.read_csv("../OlA-2/data/raw/heart_2020_cleaned.csv")
df = pd.read_csv("../OLA-2/data/raw/heart_2020_cleaned_transformed.csv")

# Assuming df is your DataFrame, X is your feature matrix and y is your target variable
correlation = df.corr()

# Get correlation between each feature and target variable
corr_target = abs(correlation["HeartDisease"])

sorted_corr = corr_target.sort_values(ascending=False)
print(sorted_corr[:100])
# the most relevant features aer: GenHealth, AgeCategory, DiffWalking, Stroke, Diabetic, PhysicalHealth
features = df[
    ["GenHealth", "AgeCategory", "DiffWalking", "Stroke", "Diabetic", "PhysicalHealth"]
]

# Handle missing values if any. For instance, dropping rows with missing values
features.dropna(inplace=True)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Elbow Method
distortions = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    distortions.append(kmeans.inertia_)

plt.plot(range(2, 10), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()

# Silhouette Score
# k = 6  # Choose the number of clusters based on the elbow diagram
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(features_scaled)
# labels = kmeans.labels_

# score = silhouette_score(features_scaled, labels)
# print(f"Silhouette Score: {score}")

# # Silhouette Plot
# from sklearn.metrics import silhouette_samples

# silhouette_vals = silhouette_samples(features_scaled, labels)

# y_ax_lower, y_ax_upper = 0, 0
# yticks = []
# for i, c in enumerate(sorted(set(labels))):
#     c_silhouette_vals = silhouette_vals[labels == c]
#     c_silhouette_vals.sort()
#     y_ax_upper += len(c_silhouette_vals)
#     color = plt.cm.nipy_spectral(float(i) / len(set(labels)))
#     plt.barh(
#         range(y_ax_lower, y_ax_upper),
#         c_silhouette_vals,
#         height=1.0,
#         edgecolor="none",
#         color=color,
#     )
#     yticks.append((y_ax_lower + y_ax_upper) / 2.0)
#     y_ax_lower += len(c_silhouette_vals)

# silhouette_avg = np.mean(silhouette_vals)
# plt.axvline(silhouette_avg, color="red", linestyle="--")

# plt.yticks(yticks, [str(c) for c in sorted(set(labels))])
# plt.ylabel("Cluster")
# plt.xlabel("Silhouette score")
# plt.show()

# num_clusters = 6

# # Apply KMeans
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# kmeans.fit(features_scaled)

# # Get the cluster labels for each data point
# labels = kmeans.labels_

# # Add the labels to the original DataFrame
# df['cluster'] = labels

# # Print the first few rows of the DataFrame to see the cluster labels
# print(df.head())

# # Visualizing the clusters is a bit tricky when you have more than two features.
# # One common approach is to use a pairplot, which shows scatter plots for each pair of features.
# # However, this requires all features to be numerical. If you have categorical features,
# # you'll need to handle them appropriately (e.g., by using one-hot encoding).
# # Here's an example of how you might create a pairplot using seaborn:

# import seaborn as sns
# sns.pairplot(df[['GenHealth', 'AgeCategory', 'DiffWalking', 'Stroke', 'Diabetic', 'cluster']], hue='cluster')
# plt.show()


from sklearn.cluster import KMeans

# Choose the number of clusters
num_clusters = 6

# Apply KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features_scaled)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the labels to the original DataFrame
df["cluster"] = labels

# Print the first few rows of the DataFrame to see the cluster labels
df.head()

# Calculate average value of each feature for each cluster
for cluster in set(labels):
    cluster_data = features[labels == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_data.mean())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add cluster labels as a new feature
df["cluster"] = labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("HeartDisease", axis=1), df["HeartDisease"], test_size=0.2, random_state=42
)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


### too computational intensive
# from sklearn.cluster import DBSCAN

# # Apply DBSCAN
# # eps is the maximum distance between two samples for them to be considered as in the same neighborhood
# # min_samples is the minimum number of samples in a neighborhood for a point to be considered as a core point
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan.fit(features_scaled)

# # Get the cluster labels for each data point
# labels = dbscan.labels_

# # Add the labels to the original DataFrame
# df["cluster"] = labels

# # Print the first few rows of the DataFrame to see the cluster labels
# print(df.head())
>>>>>>> 0fc4bcdb37b3d09dab569ccf7e1b537feb1ab21a
