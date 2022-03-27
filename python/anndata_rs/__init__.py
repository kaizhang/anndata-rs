import anndata_rs._anndata as internal
from scipy.sparse import spmatrix

class AnnData:
    def __init__(
        self,
        *,
        filename: str = None,
        X = None,
        n_obs: int = None,
        n_var: int = None,
        obs = None,
        var = None,
        pyanndata = None,
    ):
        if pyanndata is None:
            if X is not None: (n_obs, n_var) = X.shape
            self._anndata = internal.PyAnnData(filename, n_obs, n_var)
            if X is not None: self.X = X
        else:
            self._anndata = pyanndata

    @property
    def X(self): return Elem2dView(self._anndata.get_x())

    @X.setter
    def X(self, X):
        self._anndata.set_x(X)
        if isinstance(X, spmatrix):
            ...
        else:
            ...

    @property
    def obs(self): return Elem2dView(self._anndata.get_obs())

    @property
    def var(self): return Elem2dView(self._anndata.get_var())

    @property
    def obsm(self):
        return OBSM(self._anndata)

    @property
    def varm(self):
        varm = self._anndata.get_varm()
        for k in varm: varm[k] = Elem2dView(varm[k])
        return varm

    def __getitem__(self, index):
        ifnone = lambda a, b: b if a is None else a
        if index == ...:
            return self
        elif isinstance(index, slice):
            if index.stop is None:
                pass
                # do something with itertools.count()
            else:
                idx = list(range(ifnone(index.start, 0), index.stop, ifnone(index.step, 1)))
                return AnnData(pyanndata=self._anndata.subset_rows(idx))
        else:
            return AnnData(pyanndata=self._anndata.subset_rows(list(index)))

    def write(self, filename: str):
        self._anndata.write(filename)

class OBSM:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return Elem2dView(self._anndata.get_obsm(key))

    def __setitem__(self, key, data):
        self._anndata.add_obsm(key, data)

class Elem2dView:
    def __init__(self, elem):
        self._elem = elem

    def __getitem__(self, subscript):
        if subscript == ...:
            return self._elem.get_data()
        elif isinstance(subscript, slice):
            raise NotImplementedError("slice")
            # do your handling for a slice object:
            #print(subscript.start, subscript.stop, subscript.step)
            # Do your handling for a plain index
        else:
            print(subscript)

def read_h5ad(filename: str, mode: str = "r") -> AnnData:
    return AnnData(pyanndata=internal.read_anndata(filename, mode))