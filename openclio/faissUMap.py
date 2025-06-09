# faiss_umap.py  ── self-contained helper, Python ≥ 3.9
# ---------------------------------------------------------------------
import numpy as np
import faiss
import umap
from umap.umap_ import fuzzy_simplicial_set, simplicial_set_embedding, find_ab_params
from tqdm import tqdm
import inspect
from typing import Any, Optional, Dict
class ProgressWriter:
    def write(self, text):
        match = re.search(r"(\d+)/(\d+)", text)
        if match:
            n, total = map(int, match.groups())
            print("custom progress", n, total)
            # custom reporting logic here

    def flush(self):
        pass

tqdm_kwds = {"file": progress_writer", disable": False }


class ProgressUMap:
    """
    Thin convenience wrapper around umap.UMAP that shows a tqdm progress bar.

    Parameters
    ----------
    * All keyword arguments accepted by umap.UMAP can be passed straight in.
    * callback_every_epochs : int, default 1
        How many epochs to advance before the bar is updated.
        Increase this if the bar feels “laggy” on very large data sets.
    * tqdm_kw : dict, optional
        Extra keyword arguments forwarded to `tqdm(...)`
        (e.g. `{"desc": "My UMAP", "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}"}`).
    """

    def __init__(
        self,
        *args: Any,
        callback_every_epochs: int = 1,
        tqdm_kw: Optional[Dict[str, Any]] = None,
        **umap_kw: Any,
    ):
        self._callback_every_epochs = max(1, int(callback_every_epochs))
        self._tqdm_kw = {"desc": "UMAP optimise", "leave": True, **(tqdm_kw or {})}

        # Let umap-learn do the heavy lifting
        self._reducer = umap.UMAP(
            *args,
            callback=self._make_callback(),
            callback_every_epochs=self._callback_every_epochs,
            **umap_kw,
        )

        # This attribute is created lazily inside the callback
        self._pbar: Optional[tqdm] = None

    # -----------------------------------------------------------------
    # Public API: behave (almost) like umap.UMAP
    # -----------------------------------------------------------------
    def fit(self, X, y=None):
        self._reducer.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._reducer.fit_transform(X, y)

    def transform(self, X):
        return self._reducer.transform(X)

    # expose common attributes
    @property
    def embedding_(self):
        return self._reducer.embedding_

    @property
    def graph_(self):
        return self._reducer.graph_

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _make_callback(self):
        """Create the function that tqdm will call every N epochs."""

        def _callback(epoch: int, n_epochs: int, _embedding):
            # First time we enter → create the bar (know n_epochs only here)
            if self._pbar is None:
                self._pbar = tqdm(total=n_epochs, **self._tqdm_kw)

            # Advance
            self._pbar.update(self._callback_every_epochs)

            # Close the bar when we are done
            if epoch + self._callback_every_epochs >= n_epochs:
                self._pbar.close()

        return _callback

def make_pbar():
    bar = tqdm(total=reducer.n_epochs, desc="UMAP optimise")
    def cb(epoch, n_epochs, embedding):
        bar.update(1)
        if epoch + 1 == n_epochs:
            bar.close()
    return cb

def _as_float32(x):
    return np.ascontiguousarray(x, dtype="float32")

def add_with_progress(index, xb, batch_size=10_000, desc="Adding vectors"):
    """
    Add a NumPy array `xb` (n, d) to a FAISS index in batches, showing a tqdm bar.
    """
    xb = np.ascontiguousarray(xb, dtype="float32")
    n = xb.shape[0]
    for i in tqdm(range(0, n, batch_size), desc=desc):
        index.add(xb[i:i + batch_size])

def search_with_progress(index, xq, k, batch_size=4_096, desc="FAISS search"):
    """
    Run `index.search` in minibatches and return (D, I) like the original call.
    Suitable for large 'all-pairs' k-NN (query set == database).
    """
    xq = np.ascontiguousarray(xq, dtype="float32")
    n = xq.shape[0]
    D = np.empty((n, k), dtype="float32")
    I = np.empty((n, k), dtype="int64")

    for i in tqdm(range(0, n, batch_size), desc=desc):
        j = i + batch_size
        D[i:j], I[i:j] = index.search(xq[i:j], k)

    return D, I

def _faiss_knn(
        x: np.ndarray,
        k: int,
        metric: str = "cosine",
        use_hnsw: bool = True,
        M: int = 32,
        ef_c: int = 200,
        ef_s: int = 128,
        n_jobs: int = -1,
        verbose: bool = True
):
    """Return (indices, distances) of the k nearest neighbours for every point."""
    x = _as_float32(x)
    n, d = x.shape
    faiss.omp_set_num_threads(
        faiss.omp_get_max_threads() if n_jobs in (-1, None) else max(1, n_jobs)
    )

    # ----------------------- build index -------------------------------
    if metric == "cosine":
        faiss.normalize_L2(x)                                # in-place
        ip_metric = faiss.METRIC_INNER_PRODUCT
        index = (faiss.IndexFlatIP(d) if not use_hnsw
                 else faiss.IndexHNSWFlat(d, M, ip_metric))
    elif metric == "euclidean":
        l2_metric = faiss.METRIC_L2
        index = (faiss.IndexFlatL2(d) if not use_hnsw
                 else faiss.IndexHNSWFlat(d, M, l2_metric))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    if use_hnsw:
        index.hnsw.efConstruction = max(ef_c, 2 * M)
        index.hnsw.efSearch = ef_s
    if verbose:
        add_with_progress(index, x)
    else:
        index.add(x)
    

    # ---------------------- search -------------------------------------
    if verbose:
        D, I = search_with_progress(index, x, k + 1)
    else:
        D, I = index.search(x, k + 1)          # +1 -> includes self
    I, D = I[:, 1:], D[:, 1:]              # strip self-match

    if metric == "cosine":                 # convert similarity → distance
        D = 1.0 - D
    return I.astype(np.int64), _as_float32(D)


class FaissUMap:
    """
    Drop-in replacement for umap.UMAP that uses FAISS for neighbour search.

    Parameters
    ----------
    * All keyword arguments of umap.UMAP are accepted.
    * FAISS-specific kwargs (prefixed with `faiss_`):
        faiss_use_hnsw : bool  (default True)
        faiss_M        : int   (default 32)
        faiss_ef_c     : int   (default 200)
        faiss_ef_s     : int   (default 128)
        faiss_n_jobs   : int   (default -1)
    """

    def __init__(self, *args,
                 faiss_use_hnsw: bool = True,
                 faiss_M: int = 32,
                 faiss_ef_c: int = 200,
                 faiss_ef_s: int = 128,
                 faiss_n_jobs: int = -1,
                 **kwargs):
        self._umap_params = dict(kwargs)
        self._umap_params.setdefault("metric", "cosine")
        self._umap_params.setdefault("n_neighbors", 15)
        self._umap_params.setdefault("verbose", False)

        self.faiss_cfg = dict(
            use_hnsw=faiss_use_hnsw,
            M=faiss_M,
            ef_c=faiss_ef_c,
            ef_s=faiss_ef_s,
            n_jobs=faiss_n_jobs
        )
        self.embedding_ = None
        self.graph_ = None

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = _as_float32(X)
        nns = self._umap_params["n_neighbors"]
        metric = self._umap_params["metric"]
        verbose = self._umap_params['verbose']

        if verbose: print("Making FAISS K-NN")
        # 1) k-NN with FAISS -------------------------------------------------
        '''
        knn_idx, knn_dist = _faiss_knn(
            X, k=nns, metric=metric, verbose=verbose, **self.faiss_cfg
        )
        '''

        # 2) build fuzzy simplicial set ------------------------------------
        if verbose: print("Making Fuzzy Simplicial Set")
        random_state = np.random.RandomState(
            self._umap_params.get("random_state", None)
        )
        self.graph_, sigmas, rhos = fuzzy_simplicial_set(
            X=X,
            n_neighbors=nns,
            random_state=random_state,
            metric="cosine",
            #knn_indices=knn_idx,
            #knn_dists=knn_dist,
            angular=False,
            verbose=verbose
        )

        def make_pbar():
            bar = tqdm(total=reducer.n_epochs, desc="UMAP optimise")
            def cb(epoch, n_epochs, embedding):
                bar.update(1)
                if epoch + 1 == n_epochs:
                    bar.close()
            return cb

        # 3) layout / optimisation -------------------------------
        a, b = find_ab_params(reducer.spread, reducer.min_dist)
        init = "spectral"
        if verbose: print("Doing layout")


        sim_kw = dict(
            data=X,
            graph=self.graph_,
            n_components=reducer.n_components,
            initial_alpha=1.0,
            a=a,
            b=b,
            gamma=reducer.repulsion_strength,
            negative_sample_rate=reducer.negative_sample_rate,
            n_epochs=reducer.n_epochs,
            init="spectral",
            random_state=random_state,
            metric="euclidean",
            metric_kwds={},
            verbose=reducer.verbose,
        )

        # added in later version
        for extra in ("densmap", "densmap_kwds", "output_dens"):
            if extra in inspect.signature(simplicial_set_embedding).parameters:
                sim_kw[extra] = False if extra != "densmap_kwds" else {}

        self.embedding_, _ = simplicial_set_embedding(**sim_kw)

        return self

    # ------------------------------------------------------------------
    def fit_transform(self, X, y=None):
        if self.fit(X, y) is self:
            return self.embedding_

    # ------------------------------------------------------------------
    def transform(self, X):
        raise NotImplementedError("out-of-sample transform not supported yet")

    # Mimic umap-learn attributes --------------------------------------
    @property
    def embedding_(self):
        return self._embedding

    @embedding_.setter
    def embedding_(self, value):
        self._embedding = value