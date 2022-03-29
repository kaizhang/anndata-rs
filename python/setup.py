from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="anndata-rs",
    description='',
    url='https://kzhang.org/SnapATAC2/', 
    author='Kai Zhang',
    author_email='kai@kzhang.org',
    license='MIT',
    version="0.1.0",
    rust_extensions=[RustExtension("anndata_rs.pyanndata", binding=Binding.PyO3)],
    packages=[
        "anndata_rs",
    ],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.16.0",
        "pandas",
        "scipy>=1.4",
        "polars>=0.13",
        "pyarrow",
    ],
)
