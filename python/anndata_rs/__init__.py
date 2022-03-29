import anndata_rs._anndata as internal
from scipy.sparse import spmatrix
import pandas as pd
import polars

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
            if obs is not None: self.obs = obs
            if var is not None: self.var = var
            if obsm is not None: self.obsm = obsm
        else:
            self._anndata = pyanndata

    @property
    def n_obs(self): return self._anndata.n_obs

    @property
    def n_vars(self): return self._anndata.n_vars

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

    @obs.setter
    def obs(self, df):
        if isinstance(df, pd.DataFrame):
            df = polars.from_pandas(df)
        elif isinstance(df, dict):
            df = polars.from_dict(df)
        self._anndata.set_obs(df)

    @property
    def var(self):
        return Elem2dView(self._anndata.get_var())

    @var.setter
    def var(self, df):
        if isinstance(df, pd.DataFrame):
            df = polars.from_pandas(df)
        elif isinstance(df, dict):
            df = polars.from_dict(df)
        self._anndata.set_var(df)

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

    @property
    def uns(self):
        return UNS(self._anndata)

    @uns.setter
    def uns(self, uns):
        self._anndata.set_uns(uns)

    def subset(self, obs_indices = None, var_indices = None):
        def to_indices(x):
            ifnone = lambda a, b: b if a is None else a
            if isinstance(x, slice):
                if x.stop is None:
                    pass
                    # do something with itertools.count()
                else:
                    return list(range(ifnone(x.start, 0), x.stop, ifnone(x.step, 1)))
            else:
                return x
 
        self._anndata.subset_rows(to_indices(obs_indices))

    def __repr__(self) -> str:
        descr = f"AnnData object with n_obs x n_vars = {self.n_obs} x {self.n_vars}"
        if self.obs is not None: descr += f"\n    obs: {str(self.obs[...].columns)[1:-1]}"
        if self.var is not None: descr += f"\n    var: {str(self.var[...].columns)[1:-1]}"
        for attr in [
            "obsm",
            "varm",
            "uns",
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

    def __repr__(self) -> str:
        return f"AxisArrays with keys:\n{self.keys()[1:-1]}" 

class VARM:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return Elem2dView(self._anndata.get_varm(key))

    def __setitem__(self, key, data):
        self._anndata.add_varm(key, data)

    def keys(self):
        return self._anndata.list_varm()

    def __repr__(self) -> str:
        return f"AxisArrays with keys:\n{self.keys()[1:-1]}" 

class UNS:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return Elem(self._anndata.get_uns(key))

    def __setitem__(self, key, data):
        self._anndata.add_uns(key, data)

    def keys(self):
        return self._anndata.list_uns()

    def __repr__(self) -> str:
        return f"Dict with keys:\n{self.keys()[1:-1]}" 

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

class Elem:
    def __new__(cls, elem, *args, **kwargs):
        if elem is not None:
            return(super(Elem, cls).__new__(cls, *args, **kwargs))
        else:
            return None

    def __init__(self, elem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elem = elem

    def __getitem__(self, subscript):
        if subscript == ...:
            return self._elem.get_data()
        else:
            raise NameError("Please use '...' to retrieve value")


class DataFrameElem:
    def __new__(cls, elem, *args, **kwargs):
        if elem is not None:
            return(super(DataFrameElem, cls).__new__(cls, *args, **kwargs))
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