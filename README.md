Task-02
Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.
Dataset :- https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

# K-means Clustering for Customer Segmentation

This code performs K-means clustering on a dataset containing customer information to segment customers based on their purchase behavior.

## 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

2. Loading the Dataset
   The dataset containing customer information is loaded using Pandas read_csv function. The unnecessary columns, namely "CustomerID", "Gender", and "Age" are dropped.

```python
data = pd.read_csv("D:\\Projects\\Krutva_Patel\\PRODIGY_ML_02\\Mall_Customers.csv")
data.drop(["CustomerID", "Gender", "Age"], axis=1, inplace=True)
```

3. Elbow Method to Determine the Optimal Number of Clusters

The Elbow method is utilized to find the optimal number of clusters for K-means clustering. This involves running K-means with different numbers of clusters and plotting the within-cluster sum of squares (WCSS) against the number of clusters.

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
```

4. Applying K-means Clustering
   Once the optimal number of clusters is determined (in this case, 5 clusters), K-means clustering is applied to the dataset.

```python
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data)
data["Cluster"] = clusters
```

5. Printing Groups
   The customers are grouped based on the clusters assigned by K-means clustering, and the groups along with their respective customers are printed.

```python

for i in range(5):
    print("Group", i)
    print(data[data["Cluster"] == i])
    print("=" * 50)
```

6. Visualizing the Clusters
   Finally, the clusters are visualized using a scatter plot, where the x-axis represents "Annual Income (k$)" and the y-axis represents "Spending Score (1-100)".

```python
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.5)
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
```

This code provides insights into customer segmentation based on their purchase behavior, aiding in targeted marketing strategies.
