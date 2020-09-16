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