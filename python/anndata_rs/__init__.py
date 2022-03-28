import anndata_rs._anndata as internal
from scipy.sparse import spmatrix

class AnnData:
    def __init__(
        self,
        *,
        filename: str = None,
        X = None,
        n_obs: int = None,
        n_vars: int = None,
        obs = None,
        var = None,
        obsm = None,
        pyanndata = None,
    ):
        if pyanndata is None:
            if X is not None: (n_obs, n_vars) = X.shape
            self._anndata = internal.PyAnnData(filename, n_obs, n_vars)
            if X is not None: self.X = X
            if obsm is not None: self.obsm = obsm
        else:
            self._anndata = pyanndata
        self.n_obs = self._anndata.n_obs
        self.n_vars = self._anndata.n_vars

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
    def obs(self): 
        return Elem2dView(self._anndata.get_obs())

    @property
    def var(self):
        return Elem2dView(self._anndata.get_var())

    @property
    def obsm(self):
        return OBSM(self._anndata)

    @obsm.setter
    def obsm(self, obsm):
        self._anndata.set_obsm(obsm)

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

    def __repr__(self) -> str:
        descr = f"AnnData object with n_obs x n_vars = {self.n_obs} x {self.n_vars}"
        if self.obs is not None: descr += f"\n    obs: {str(self.obs[...].columns)[1:-1]}"
        if self.var is not None: descr += f"\n    var: {str(self.var[...].columns)[1:-1]}"
        for attr in [
            "obsm",
            "varm",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def write(self, filename: str):
        self._anndata.write(filename)

class OBSM:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return Elem2dView(self._anndata.get_obsm(key))

    def __setitem__(self, key, data):
        self._anndata.add_obsm(key, data)

    def keys(self):
        return self._anndata.list_obsm()

class Elem2dView:
    def __new__(cls, elem, *args, **kwargs):
        if elem is not None:
            return(super(Elem2dView, cls).__new__(cls, *args, **kwargs))
        else:
            return None

    def __init__(self, elem, *args, **kwargs):
        super().__init__(*args, **kwargs)
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