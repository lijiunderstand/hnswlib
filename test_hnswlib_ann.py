
from __future__ import absolute_import
import numpy as np
import os, psutil
import time
import faiss

import sys
sys.path.append("install/lib-faiss")  # noqa
import numpy
import sklearn.preprocessing
import ctypes
import hnswlib
import h5py

from multiprocessing.pool import ThreadPool
import psutil

# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.

# loc ='/home/lijie/workspace/FML/scann/google-research/scann/glove-100-angular.hdf5'    
# loc ='/data/ANN/ann-benchmarks/data/mnist-784-euclidean.hdf5'  
# loc ='/data/ANN/ann-benchmarks/data/sift-128-euclidean.hdf5'
loc ='/data/ANN/ann-benchmarks/data/deep-image-96-angular.hdf5'
# loc = '/home/lijie/workspace/ann_benchmark_all/ann-benchmarks/data/mnist-784-euclidean.hdf5'
# loc ='/mnt/intel/ANN/ann-benchmarks/data/random-s-100-angular.hdf5'
# loc = '/mnt/intel/ANN/ann-benchmarks/data/nytimes-16-angular.hdf5'
# loc ='/mnt/intel/ANN/ann-benchmarks/data/glove-200-angular.hdf5'
glove_h5py = h5py.File(loc, "r")
list(glove_h5py.keys())

dataset = glove_h5py['train']
queries = glove_h5py['test']
xb = glove_h5py['train']
xq = glove_h5py['test']
# xb.astype(np.float32)
# xb.astype(np.float32)
print(xb.shape)
print(xq.shape)
nb = xb.shape[0]
nq = xb.shape[0]
xb = np.array(xb)
xq = np.array(xq)
print('xq type',type(xq))

xb = xb / np.linalg.norm(xb, axis=1)[:, np.newaxis]
d = xb.shape[1]
print('d:', d)

class BaseANN(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X):
        pass

    def query(self, q, n):
        return []  # array of candidate indices

    def batch_query(self, X, n):
        """Provide all queries at once and let algorithm figure out
           how to handle it. Default implementation uses a ThreadPool
           to parallelize query processing."""
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name
    
    
class HnswLib(BaseANN):
    def __init__(self, metric, M, efConstruction):
        # self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.metric = metric
        # self.method_param = method_param?
        self.M = M
        self.efConstruction = efConstruction
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = 'hnswlib (%s)' % (self.M+ self.efConstruction)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(max_elements=len(X),
                          ef_construction=self.efConstruction,
                          M=self.M)
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        # self.p.set_num_threads(1)
        self.p.set_num_threads(8)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def batch_query(self, X, n):
        # print(np.expand_dims(v,axis=0).shape) #(1,10000,512)
        # print(X.shape)
        # print((self.p.knn_query(X, k=n)[0]).shape)  #(10000,10)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        self.res =  self.p.knn_query(X, k=n)[0]
    def freeIndex(self):
        del self.p

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

if __name__=="__main__":
    topk =1
    # algo = HnswLib("l2", 64, 500)
    algo = HnswLib("ip", 32, 40)
    # algo = HnswLib("l2", 32, 40)
    # algo = HnswLib("cosine", 64, 500)
    print("Index: ",algo.metric)
    print('topk', topk)
    print('M', algo.M)
    print('efconstruction', algo.efConstruction)
    
    t0 =time.time()
    memory_usage_before = algo.get_memory_usage()
    
    algo.fit(xb)
    build_time = time.time()-t0
    index_size = algo.get_memory_usage() - memory_usage_before
    print('Built index in: ', build_time)
    print('Index size: ', index_size)
    
    algo.set_query_arguments(64)

    # algo.query(xq, topk)
    # results = algo.batch_query(xq, topk)
    start = time.time()
    algo.batch_query(xq, topk)
    end = time.time()
    
    process = psutil.Process(os.getpid())
    print(process.memory_full_info())
    results = algo.get_batch_results()
    # _, neighbors=results
    print("Recall:", compute_recall(results, glove_h5py['neighbors'][:, :topk]))
    speed = (end-start)/nq
    qps =1000/speed
    print("qps:", qps)