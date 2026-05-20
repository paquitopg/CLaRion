"""
Per-module Cython setup. Kept around for backwards compatibility — the
canonical build entry point is now the project-root setup.py at
CLaRion/setup.py, which builds *all* of CLaRiON's Cython extensions
(topk on the decoder side, encoder + index on the encoder side).
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "parallel.cython_topk",
        ["src/parallel/cython_topk.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="clara_topk_cpu",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": 3}),
)