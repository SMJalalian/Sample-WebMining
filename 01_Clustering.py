import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('Exports/Customers.csv') # Load dataset of customers by some features
selected_data = data.iloc[:, [2, 3]].values # select Age and annual income ( for clustering )

#********************************************************************************

result = [] #Create empty list
for i in range(1, 50):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(selected_data)
    result.append(kmeans.inertia_)
plt.plot(range(1, 50), result)
plt.title('The Elbow Method Calculation')
plt.xlabel('Number of clusters')
plt.ylabel('within-cluster Sums of Squares')
plt.show()

# This formula illustrates number (4) will be a proper selection for cluters !!!

#********************************************************************************

# Execute K-Means to the dataset inorder to create 4 cluster
kmeans = KMeans(n_clusters = 4)
y_kmeans = kmeans.fit_predict(selected_data)

#********************************************************************************

# illustrate all nodes in plot
plt.scatter(selected_data[y_kmeans == 0, 0], selected_data[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster A')
plt.scatter(selected_data[y_kmeans == 1, 0], selected_data[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster B')
plt.scatter(selected_data[y_kmeans == 2, 0], selected_data[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster C')
plt.scatter(selected_data[y_kmeans == 3, 0], selected_data[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster D')
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()