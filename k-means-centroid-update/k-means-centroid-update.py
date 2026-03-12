from collections import defaultdict
import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    d = len(points[0])
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k

    for point, cluster_id in zip(points, assignments):
        counts[cluster_id] += 1
        for j in range(d):
            sums[cluster_id][j] += point[j]

    return [
        [sums[i][j] / counts[i] for j in range(d)] if counts[i] > 0 else [0.0] * d
        for i in range(k)
    ]
    