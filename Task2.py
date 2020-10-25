import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, jaccard_score
from sklearn.metrics.pairwise import pairwise_distances
from numpy.linalg import norm
from scipy import spatial

dataset = pd.read_csv('iris.data')
x = dataset.iloc[:, [0, 1, 2, 3]].values
y_actual = dataset.iloc[:, [4]].values
for i in range(len(y_actual)):
	if (y_actual[i] == 'Iris-setosa'):
		y_actual[i] = 0
	elif (y_actual[i] == 'Iris-versicolor'):
		y_actual[i] = 1
	else:
		y_actual[i] = 2

class Kmeans:
	def __init__(self, numClusters, maxIterations=100, distanceMetric='Euclidean'):
		self.numClusters = numClusters
		self.maxIterations = maxIterations
		self.distanceMetric = distanceMetric

	def computeCentroids(self, X, labels):
		centroids = np.zeros((self.numClusters, X.shape[1]))
		for k in range(self.numClusters):
			centroids[k, :] = np.mean(X[labels == k, :], axis=0)
		return centroids

	def computeEuclideanDistance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.numClusters))
		for k in range(self.numClusters):
			row_norm = norm(X - centroids[k, :], axis=1)
			distance[:, k] = np.square(row_norm)
		return distance

	def computeJacardDistance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.numClusters))
		for k in range(self.numClusters):
			#distance[i,k] = jaccard_score(X[i], centroids[k,:])
			for i in range(len(X)):
				value = 0.0
				for val in range(len(X[i])):
					if (X[i][val] == centroids[k,val]):
						value += 1.0
				distance[i:k] = value / 2.0
		print(distance)
		return distance

	def computeCosineDistance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.numClusters))
		for k in range(self.numClusters):
			for i in range(len(X)):
				distance[i,k] = spatial.distance.cosine(X[i], centroids[k,:])
		return distance

	def computeSSE(self, X, labels, centroids):
		distance = np.zeros(X.shape[0])
		for k in range(self.numClusters):
			distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
		return np.sum(np.square(distance))
	
	def fit(self, X):
		np.random.RandomState(123)
		random_idx = np.random.permutation(X.shape[0])
		self.centroids = X[random_idx[:self.numClusters]]

		for i in range(self.maxIterations):
			self.old_centroids = self.centroids
			if (self.distanceMetric == 'Euclidean'):
				distance = self.computeEuclideanDistance(X, self.old_centroids)
			elif (self.distanceMetric == 'Cosine'):
				distance = self.computeCosineDistance(X, self.old_centroids)
			else:
				distance = self.computeJacardDistance(X, self.old_centroids)
			self.labels = np.argmin(distance, axis=1)
			self.centroids = self.computeCentroids(X, self.labels)
			if np.all(self.old_centroids == self.centroids):
				break
		self.iterations = i
		self.error = self.computeSSE(X, self.labels, self.centroids)
	
	def predict(self, X):
		if (self.distanceMetric == 'Euclidean'):
			distance = self.computeEuclideanDistance(X, self.old_centroids)
		elif(self.distanceMetric == 'Cosine'):
			distance = self.computeCosineDistance(X, self.old_centroids)
		else:
			distance = self.computeJacardDistance(X, self.old_centroids)
		return np.argmin(distance, axis=1)

kmeans = Kmeans(numClusters=3, maxIterations=100, distanceMetric='Jaccard')
kmeans.fit(x)
centroids = kmeans.centroids
print(kmeans.error)
print(kmeans.iterations)
print(centroids)

accuracy = 0.0
length = len(y_actual)
y_pred = kmeans.predict(x)
print(y_pred)
for i in range(len(y_pred)):
	if (y_pred[i] == y_actual[i]):
		accuracy += 1
accuracy = accuracy / length
print(accuracy)

fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(x[kmeans.labels == 0, 0], x[kmeans.labels == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[kmeans.labels == 1, 0], x[kmeans.labels == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[kmeans.labels == 2, 0], x[kmeans.labels == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='yellow', label='centroid')
plt.legend()
plt.xlim([4, 8])
plt.ylim([1, 5])
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal');
plt.show()