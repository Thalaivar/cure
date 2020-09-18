import numpy as np
cimport numpy as np
from scipy.spatial.distance import pdist

cdef double dense_dist_mat_at_ij(double[:] dist, int i, int j, int n):
    cdef int idx
    if i < j:
        idx = i*n - i*(i+1) // 2 - (j-i-1)
    elif i > j:
        idx = j*n - j*(j+1) // 2 - (i-j-1)
    else:
        return 0.0

    return dist[idx]

cpdef tuple well_scattered_points(int n_rep, np.ndarray[np.double_t, ndim=1] mean, np.ndarray[np.double_t, ndim=2] data):
    cdef int n = data.shape[0]
    
    # if the cluster contains less than no. of rep points, all points are rep points
    if n <= n_rep:
        return list(data), np.arange(data.shape[0])
    
    cdef double[:] distances = pdist(data)

    cdef int idx = np.argmax(np.linalg.norm(data - mean, axis=1))
    
    # keep track of distances to scattered points as they are identified
    cdef np.ndarray[np.double_t, ndim=2] dist_to_scatter = -1.0*np.ones((n_rep, n)).astype(np.float64)

    cdef list scatter_idx = [idx]
    cdef int i, j, k, max_point, min_dist_idx
    cdef double min_dist, max_dist, dist
    
    for i in range(n_rep-1):
        max_dist = 0.0
        for j in range(n):
            # calculate distance of point to latest scatter point
            dist_to_scatter[i,j] = dense_dist_mat_at_ij(distances, scatter_idx[-1], j, n)
            # minimum distance of point from scattered points
            min_dist = np.min(dist_to_scatter[:i+1,j])
            if min_dist > max_dist:
                max_dist = min_dist
                max_point = j
        scatter_idx.append(max_point)
        
    return [data[i] for i in scatter_idx], scatter_idx