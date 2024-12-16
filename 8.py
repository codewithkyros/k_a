# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic customer data
# For simplicity, we use make_blobs to simulate customer data
# Features: Annual Income and Spending Score
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=["Annual Income", "Spending Score"])

# Step 2: Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(df["Annual Income"], df["Spending Score"], c='gray', s=50)
plt.title("Customer Data (Before Clustering)")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Step 3: Apply k-means clustering
# Determine the optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Choose the optimal number of clusters (e.g., k=4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Step 4: Visualize the clustered dataset
plt.figure(figsize=(8, 6))
plt.scatter(df["Annual Income"], df["Spending Score"], c=df["Cluster"], cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label="Centroids")
plt.title(f"Customer Clusters (k={k_optimal})")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
