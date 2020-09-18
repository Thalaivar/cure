import numpy as np
import logging
import itertools
from _heapq import *
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist, pdist
from CURE.cluster_utils import well_scattered_points

# def dense_dist_mat_at_ij(dist, i, j, n):
#     if i < j:
#         idx = int(i*n - i*(i+1) // 2 - (j-i-1))
#     elif i > j:
#         idx = int(j*n - j*(j+1) // 2 - (i-j-1))
#     else:
#         return 0.0

#     return dist[idx]

# def well_scattered_points(n_rep: int, mean: np.ndarray, data: np.ndarray):
#     n = data.shape[0]
#     # if the cluster contains less than no. of rep points, all points are rep points
#     if n <= n_rep:
#         return list(data), np.arange(data.shape[0])
    
#     # calculate distances for fast access
#     distances = pdist(data)

#     # farthest point from mean
#     idx = np.argmax(np.linalg.norm(data - mean, axis=1))
#     # get well scattered points
#     scatter_idx = [idx]
#     for _ in range(1, n_rep):
#         max_dist = 0.0
#         for j in range(n):
#             # minimum distances from points in scatter_idx
#             min_dist = min([dense_dist_mat_at_ij(distances, idx, j, n) for idx in scatter_idx])
#             if min_dist > max_dist:
#                 max_dist = min_dist
#                 max_point = j
        
#         scatter_idx.append(max_point)
    
#     return [data[i] for i in scatter_idx], scatter_idx

class cure_cluster:
    def __init__(self, data, index, n_rep, alpha, labels=None):
        if len(data.shape) == 1:
            data = data.reshape(1,-1)
        self.data = data
        self.idx = index
        self.n_rep = n_rep
        self.alpha = alpha
        self.labels = labels if labels is not None else [index]

        self.mean = None
        self.rep = []
        self.rep_idx = []

        if self.data.shape[0] == 1:
            self._calculate_mean()
            self.rep = [data]
            self.rep_idx = [index] 

        self.distance = None
        self.closest = None

    def _calculate_mean(self):
        self.mean = self.data.mean(axis=0)
    
    def __lt__(self, c):
        return self.distance < c.distance
    
    def __len__(self):
        return self.data.shape[0]
    
    def _exemplars_from_data(self):
        scattered_points, _ = well_scattered_points(self.n_rep, self.mean, self.data)
        # shrink points toward mean
        rep = [p + self.alpha*(self.mean - p) for p in scattered_points]   
        self.rep = rep

    def _unshrink_exemplars(self):
        return [(rep - self.alpha * self.mean)/(1 - self.alpha) for rep in self.rep]

class CURE:
    def __init__(self, k, n_rep=10, alpha=0.5):
        self.cluster_args = {'n_rep': n_rep, 'alpha': alpha}
        self.desired_clusters = k

        # tracks which point in tree is active
        self.active_points = None
        self.tree_ = None
        self.heap_ = None

        self.labels_ = None

    def fit(self, data):
        self._create_kdtree(data)
        self._create_heap(data)

        u = heappop(self.heap_)
        while len(self.heap_) > self.desired_clusters:
            logging.debug(f'clusters in heap: {len(self.heap_)}, desired clusters: {self.desired_clusters}')
            v = u.closest

            self.heap_.remove(v)

            self._remove_representative_points(u)
            self._remove_representative_points(v)

            w = self._merge_clusters(u, v)

            self._insert_representative_points(w)

            w.distance = np.inf
            relocate_clusters = []
            for c in self.heap_:
                dist = self._distance_between_clusters(w, c)
                if  dist < w.distance:
                    w.closest = c
                    w.distance = dist
                
                if c.closest is u or c.closest is v:
                    if c.distance > dist:
                        c.closest = w
                        c.distance = dist
                    else:
                        (closest_cluster, closest_dist) = self._closest_cluster(c, dist)
                        
                        if closest_cluster is None or closest_cluster in w.rep_idx:
                            c.closest = w
                            c.distance = dist
                        
                        else:
                            c.closest = self._find_cluster_with_rep_idx(closest_cluster)
                            c.distance = closest_dist

                    relocate_clusters.append(c)

                elif c.distance > dist:
                    c.closest = w
                    c.distance = dist
                    relocate_clusters.append(c)
            
            for c in relocate_clusters:
                self.heap_.remove(c)
                heappush(self.heap_, c)
                
            u = heappushpop(self.heap_, w)

        # self._create_assignments()
    
    def _create_heap(self, data):
        self.heap_ = [cure_cluster(data[i], i, **self.cluster_args) for i in range(data.shape[0])]
        
        # calculate representative points
        for c in self.heap_:
            c._exemplars_from_data()

        # for each point find nearest neighbor
        for i in range(len(self.heap_)):
            d, idx = self.tree_.query(data[i].reshape(1,-1), k=2)
            d, idx = d[0][1], idx[0][1]
            self.heap_[i].distance = d
            self.heap_[i].closest = self.heap_[idx]
        
        heapify(self.heap_)

    def _create_kdtree(self, data):
        self.tree_ = KDTree(data)
        # initially, every point is active in tree
        self.active_points = np.ones((data.shape[0],), dtype=np.int8)
    
    def _distance_between_clusters(self, u: cure_cluster, v:cure_cluster):        
        u.rep, v.rep = np.array(u.rep), np.array(v.rep)
        distances = cdist(u.rep, v.rep)
        return np.min(distances)


    def _remove_representative_points(self, cluster: cure_cluster):
        for i in cluster.rep_idx:
            # deactivate point in tree
            self.active_points[i] = -1
    
    def _insert_representative_points(self, cluster: cure_cluster):
        for i in cluster.rep_idx:
            self.active_points[i] = 1

    def _closest_cluster(self, cluster: cure_cluster, distance: float):
        min_dist = np.inf

        for p in cluster.rep:
            # get all points within distance
            idx, dist = self.tree_.query_radius(p.reshape(1,-1), distance, return_distance=True)
            idx, dist = idx[0], dist[0]

            # remove invalid neighbors
            valid_points = []
            for i in range(idx.size):
                # if distance is zero, the neighbor is the same as query
                if dist[i] == 0.0:
                    continue
                # point is in query point's cluster
                if idx[i] in cluster.rep_idx:
                    continue
                # point is deleted from kdtree
                if self.active_points[idx[i]] < 0:
                    continue
                valid_points.append(i)
            idx, dist = idx[valid_points], dist[valid_points]

            if idx.size == 0:
                # no points within distance
                return (None, None)
            else:
                # closest point
                closest_idx = np.argmin(dist)
                idx, dist = idx[closest_idx], dist[closest_idx]

                if dist < min_dist:
                    min_dist = dist
                    closest_rep_idx = idx

        return (closest_rep_idx, min_dist)

    def _find_cluster_with_rep_idx(self, rep_idx):
        cluster = None
        for c in self.heap_:
            if rep_idx in c.rep_idx:
                cluster = c
                break

        return cluster

    def _merge_clusters(self, u: cure_cluster, v: cure_cluster):
        combined_data = np.vstack((u.data, v.data))
        w = cure_cluster(combined_data, u.idx+v.idx, **self.cluster_args)
        w.labels = u.labels + v.labels

        # get new mean
        w.mean = (len(u)*u.mean + len(v)*v.mean)/(len(u) + len(v))

        # get "well scattered" points from both clusters
        scattered_points = u._unshrink_exemplars() + v._unshrink_exemplars()
        new_rep_points, retained_idx = well_scattered_points(w.n_rep, w.mean, np.array(scattered_points))
            
        # get representative points for merged cluster by shrinking
        w.rep = np.array([p + w.alpha*(w.mean - p) for p in new_rep_points])
        # get indices of representative points from u, v
        rep_idx = u.rep_idx + v.rep_idx
        w.rep_idx = list(np.array(rep_idx)[retained_idx])

        return w
