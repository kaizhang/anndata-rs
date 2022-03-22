import anndata_rs._anndata as internal

class AnnData:
    def __init__(self, pyanndata):
        self._anndata = pyanndata

    @property
    def X(self): return Elem2dView(self._anndata.get_x())

    @property
    def obs(self): return Elem2dView(self._anndata.get_obs())

    @property
    def var(self): return Elem2dView(self._anndata.get_var())

    @property
    def obsm(self):
        obsm = self._anndata.get_obsm()
        for k in obsm: obsm[k] = Elem2dView(obsm[k])
        return obsm

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
                return AnnData(self._anndata.subset_rows(idx))
        else:
            return AnnData(self._anndata.subset_rows(list(index)))

    def write(self, filename: str):
        self._anndata.write(filename)

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
    return AnnData(internal.read_anndata(filename, mode))