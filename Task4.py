import math

clusterAPoints = [(4.7, 3.2), (4.9, 3.1), (5.0, 3.0), (4.6, 2.9)]
clusterBPoints = [(5.9, 3.2), (6.7, 3.1), (6.0, 3.0), (6.2, 2.8)]

distanceBetweenPairs = []
avgDist = []
for a in clusterAPoints:
	totalDistance = 0
	for b in clusterBPoints:
		distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))
		totalDistance += distance
		distanceBetweenPairs.append(distance)
	avgDist.append(totalDistance / 4)

print(distanceBetweenPairs)
print(avgDist)

dist = 0
for i in avgDist:
	dist += i

print(dist / 4)