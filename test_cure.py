import numpy as np
from tqdm import tqdm
from sklearn import datasets
from BSOID.cure import CURE
from pyclustering.cluster.cure import cure as pyCURE
import logging
import time
logging.basicConfig(level=logging.INFO)

dims = [10,20,40]

times = []
for i in tqdm(range(len(dims))):
    dim = dims[i]
    data, _ = datasets.make_blobs(n_samples=int(5e3), n_features=dim, centers=25)

    # my
    cure = CURE(25, 100, 100, n_rep=10, alpha=0.5)
    start = time.time()
    cure.process(data)
    my = time.time() - start

    # pyclustering python
    pycure = pyCURE(data, 25, 10, ccore=False)
    start = time.time()
    pycure.process()
    pyc = time.time() - start

    times.append([my, pyc])

import joblib
with open('clustering_benchmark.sav', 'wb') as f:
    joblib.dump(times, f)