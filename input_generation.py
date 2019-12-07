import numpy as np
def randomMetricGraph(numOfVertices, lowerLimit, upperLimit, edgeProb):
	X = np.random.uniform(lowerLimit, upperLimit, numOfVertices)
	Y = np.random.uniform(lowerLimit, upperLimit, numOfVertices)
	G = np.empty((numOfVertices, numOfVertices))
	for i in range(numOfVertices):
		for j in range(i+1):
			a = np.array((X[i], Y[i]))
			b = np.array((X[j], Y[j]))
			G[i, j] = round(np.linalg.norm(a-b), 5) if np.random.uniform() < edgeProb else 0
			G[j, i] = G[i, j]
	return '\n'.join( ( ' '.join(str(d) if d>0 else 'x' for d in row) for row in G ) ) + '\n'