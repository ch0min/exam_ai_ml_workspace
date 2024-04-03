import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
k = 6  # Choose the number of clusters based on the elbow diagram
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(features_scaled)
labels = kmeans.labels_

score = silhouette_score(features_scaled, labels)
print(f"Silhouette Score: {score}")

# Silhouette Plot
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(features_scaled, labels)

y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(sorted(set(labels))):
    c_silhouette_vals = silhouette_vals[labels == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = plt.cm.nipy_spectral(float(i) / len(set(labels)))
    plt.barh(
        range(y_ax_lower, y_ax_upper),
        c_silhouette_vals,
        height=1.0,
        edgecolor="none",
        color=color,
    )
    yticks.append((y_ax_lower + y_ax_upper) / 2.0)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")

plt.yticks(yticks, [str(c) for c in sorted(set(labels))])
plt.ylabel("Cluster")
plt.xlabel("Silhouette score")
plt.show()

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
