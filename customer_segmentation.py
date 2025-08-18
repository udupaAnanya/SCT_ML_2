import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Quick look at the data
print("First 5 rows of dataset:")
print(df.head(), "\n")

# Step 3: Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow method + Silhouette scores
inertia = []
silhouette_scores = {}

for k in range(2, 11):  # silhouette not defined for k=1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores[k] = silhouette_score(X_scaled, labels)

# Plot Elbow Method
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Print Silhouette Scores
print("\nSilhouette Scores:")
for k, score in silhouette_scores.items():
    print(f"k={k}, silhouette={score:.3f}")

# Step 6: Train final KMeans model (say k=5, or choose based on scores)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 7: Visualize clusters with seaborn
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], 
                hue=df['Cluster'], palette="viridis", s=60, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation (K-means)")
plt.legend()
plt.show()

# Step 8: Detailed Cluster Profiling
cluster_summary = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean','min','max'],
    'Spending Score (1-100)': ['mean','min','max'],
    'CustomerID': 'count'
})
print("\nCluster Analysis (mean, min, max, count):")
print(cluster_summary)

# Step 9: Export results
df.to_csv("Mall_Customers_Segmented.csv", index=False)
print("\nClustered dataset saved as 'Mall_Customers_Segmented.csv'")
