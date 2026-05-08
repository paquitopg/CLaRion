from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "parallel.cython_topk",
        ["src/parallel/cython_topk.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-fopenmp"
        ],
        extra_link_args=[
            "-fopenmp"
        ]
    )
]

setup(
    name="clara_topk_cpu",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": 3
        }
    )
)