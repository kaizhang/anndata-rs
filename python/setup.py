from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="anndata_rs",
    description='',
    url='https://github.com/kaizhang/anndata-rs', 
    author='Kai Zhang',
    author_email='kai@kzhang.org',
    license='MIT',
    version="0.2.0",
    rust_extensions=[RustExtension("_anndata_rs", binding=Binding.PyO3)],
    packages=["anndata_rs"],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "anndata>=0.8",
        "numpy>=1.16",
        "pandas",
        "scipy>=1.4",
        "polars>=0.14",
        "pyarrow",
    ],
)