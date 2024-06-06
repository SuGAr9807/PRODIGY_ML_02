import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Drop unnecessary columns
data.drop(["CustomerID", "Gender", "Age"], axis=1, inplace=True)

# Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)


# Applying k-means to the dataset
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data)

# Add clusters to the dataset
data["Cluster"] = clusters
for i in range(5):
    print("Group", i)
    print(data[data["Cluster"] == i])
    print("=" * 50)
# Visualizing the clusters
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.5)
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
