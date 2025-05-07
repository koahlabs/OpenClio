import numpy as np
import faiss

class FaissKMeans:
    """
    A minimalist scikit-learn-like wrapper around faiss.Kmeans.

    Parameters
    ----------
    n_clusters : int, default=8
    max_iter   : int, default=25
    n_init     : int, default=1          (-> faiss `nredo`)
    tol        : float, default=1e-4     (ignored by faiss, for API parity)
    approximate: bool, default=False     If True uses HNSW instead of brute-force
    M, ef      : int, HNSW construction / search params (only if approximate)
    random_state : int or None
    n_jobs     : int, default=-1         (-1 = use all cores)
    verbose    : bool / int
    """

    def __init__(self, n_clusters=8, max_iter=25, n_init=1,
                 tol=1e-4, approximate=False, M=32, ef=256,
                 random_state=None, n_jobs=-1, verbose=False):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.n_init       = n_init
        self.tol          = tol
        self.approximate  = approximate
        self.M, self.ef   = M, ef
        self.random_state = random_state
        self.n_jobs       = n_jobs
        self.verbose      = verbose

    @staticmethod
    def _as_float32(x):
        """make sure x is contiguous float32"""
        x = np.ascontiguousarray(x, dtype='float32')
        return x

    def _make_index(self, d):
        """return a faiss index according to approximate flag"""
        if self.approximate:
            index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch       = self.ef
            index.hnsw.efConstruction = max(200, self.ef)
        else:
            index = faiss.IndexFlatL2(d, faiss.METRIC_INNER_PRODUCT)
        return index

    def fit(self, X, y=None):
        X = self._as_float32(X)
        n_samples, d = X.shape

        faiss.omp_set_num_threads(
            faiss.omp_get_max_threads() if self.n_jobs in (-1, None)
            else max(1, self.n_jobs)
        )

        # build kmeans object
        km = faiss.Kmeans(
            d=d,
            k=self.n_clusters,
            niter=self.max_iter,
            nredo=self.n_init,
            verbose=bool(self.verbose),
            spherical=True,  # cosine similiarity
            seed=self.random_state,
        )
        km.cp.min_points_per_centroid = 1 # remove warning

        # plug ANN index if requested
        if self.approximate:
            km.index = self._make_index(d)        
        
        km.train(X)

        self.cluster_centers_ = (
            km.centroids
            .reshape(self.n_clusters, d)
            .copy()
        )

        # labels & inertia
        D, I = km.index.search(X, 1)
        self.labels_   = I.ravel().astype(np.int64, copy=True)
        self.inertia_  = float(D.sum())

        return self