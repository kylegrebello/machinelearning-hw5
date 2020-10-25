import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy.linalg import norm

dataset = pd.read_csv('data.csv')
dataset = dataset.drop('Team', axis=1)
xAxis = dataset.iloc[:, 0].values
yAxis = dataset.iloc[:, 1].values
data = (xAxis, yAxis)
#print(dataset)

centroids = [(4,6), (5,4)]
#centroids = [(3,3), (8,3)]
#centroids = [(3,2), (4,8)]

def getcentroidListsManhattan(centroids):
	centroid1List = []
	centroid2List = []
	for i in range(len(xAxis)):
		distanceToFirstcentroid = abs(xAxis[i] - centroids[0][0]) + abs(yAxis[i] - centroids[0][1])
		distanceToSecondcentroid = abs(xAxis[i] - centroids[1][0]) + abs(yAxis[i] - centroids[1][1])
		if (distanceToFirstcentroid < distanceToSecondcentroid):
			centroid1List.append(i)
		else:
			centroid2List.append(i)
	return centroid1List, centroid2List

def getcentroidLists(centroids):
	centroid1List = []
	centroid2List = []
	for i in range(len(xAxis)):
		distanceToFirstcentroid = math.sqrt(math.pow(xAxis[i] - centroids[0][0], 2) + math.pow(yAxis[i] - centroids[0][1], 2))
		distanceToSecondcentroid = math.sqrt(math.pow(xAxis[i] - centroids[1][0], 2) + math.pow(yAxis[i] - centroids[1][1], 2))
		if (distanceToFirstcentroid < distanceToSecondcentroid):
			centroid1List.append(i)
		else:
			centroid2List.append(i)
	return centroid1List, centroid2List

def getNewcentroids(centroidLists):
	newcentroids = []
	for i in range(len(centroidLists)):
		listSize = len(centroidLists[i])
		xTotal = 0
		yTotal = 0
		for j in centroidLists[i]:
			xTotal += xAxis[j]
			yTotal += yAxis[j]
		newcentroids.append((xTotal / listSize, yTotal / listSize))
	return newcentroids

centroidList = []

maxIterations = 100
currentIteration = 0
while (currentIteration < 100):
	currentIteration += 1
	centroidList.append(centroids)
	centroid1List, centroid2List = getcentroidLists(centroids)
	#centroid1List, centroid2List = getcentroidListsManhattan(centroids)
	newcentroids = getNewcentroids((centroid1List, centroid2List))
	if (newcentroids == centroids):
		break
	centroids = newcentroids

centroidXs = []
centroidYs = []
for i in range(len(centroids)):
	centroidXs.append(centroids[i][0])
	centroidYs.append(centroids[i][1])

centroid1ListXs = []
centroid1ListYs = []
for i in centroid1List:
	centroid1ListXs.append(xAxis[i])
	centroid1ListYs.append(yAxis[i])

centroid2ListXs = []
centroid2ListYs = []
for i in centroid2List:
	centroid2ListXs.append(xAxis[i])
	centroid2ListYs.append(yAxis[i])


print(centroids)
plt.figure(figsize=(6, 6))
plt.scatter(centroid1ListXs, centroid1ListYs, c='green', label='cluster1')
plt.scatter(centroid2ListXs, centroid2ListYs, c='blue', label='cluster2')
plt.scatter(centroidXs, centroidYs, marker='*', s=300, c='r', label='centroid')
plt.xlabel('wins in 2016')
plt.ylabel('wins in 2017')
plt.title('Visualization of raw data');

plt.show()