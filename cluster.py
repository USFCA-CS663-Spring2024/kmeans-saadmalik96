import random
import numpy as np

class cluster:
    def __init__(self, k=5, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
                
    def fit(self, X):
        # Place k centroids randomly
        indices = random.sample(range(len(X)), self.k)
        self.centroids = []
        for i in indices:
            self.centroids.append(X[i])

        # Repeat to convergence or max_iterations
        for i in range(self.max_iterations):
            
            # Create k empty clusters
            clusters = []
            for j in range(self.k):
                clusters.append([])
            
            # Assigment 
            for x in X:
                closest_centroid = self.get_closest_centroid(x)
                clusters[closest_centroid].append(x)

            # Update
            new_centroids = []
            for cluster in clusters:
                    new_centroid = np.mean(cluster, axis=0)
                    new_centroids.append(new_centroid.tolist())
            
            if np.allclose(new_centroids, self.centroids, atol=1e-6):
                break
            else:
                self.centroids = new_centroids
        
        # generate labels and clusters
        labels = []
        for x in X:
            closest_centroid = self.get_closest_centroid(x)
            labels.append(closest_centroid)
        
        return labels, self.centroids
    
    def get_closest_centroid(self, x):
        closest_index = 0
        min_distance = float('inf')
        for i, centroid in enumerate(self.centroids):
            distance = np.linalg.norm(np.array(x) - np.array(centroid))
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index
    
# cluster = cluster(2)
# X = [ [0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10] ]
# clusters, centroids = cluster.fit(X)
# print(clusters)
# print(centroids)