"""
CLaRiON project-root setup.py.

Builds every Cython extension used by the encoder + index + decoder modules.

Run from the project root:

    python setup.py build_ext --inplace

The extensions land alongside their .pyx sources under src/parallel/ so that
`from src.parallel import cython_encoder, cython_index, cython_topk` works
both in the source tree and from any working directory that has src/ on PYTHONPATH.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import subprocess

# OpenMP flags differ between toolchains.
EXTRA_COMPILE = ["-O3", "-march=native",  "-ffast-math"]
EXTRA_LINK: list[str] = []


def _brew_prefix(formula: str, default: str) -> str:
    """Find a Homebrew formula prefix, falling back to a sane default."""
    try:
        out = subprocess.check_output(
            ["brew", "--prefix", formula], stderr=subprocess.DEVNULL
        ).decode().strip()
        if out:
            return out
    except Exception:
        pass
    return default


if platform.system() == "Darwin":
    # Apple clang doesn't ship OpenMP. The user should:
    #     brew install libomp
    # We then locate the prefix and add include/lib paths so clang finds it.
    libomp_prefix = _brew_prefix(
        "libomp",
        "/opt/homebrew/opt/libomp" if platform.machine() == "arm64"
        else "/usr/local/opt/libomp",
    )
    EXTRA_COMPILE += [
        "-Xpreprocessor", "-fopenmp",
        f"-I{libomp_prefix}/include",
    ]
    EXTRA_LINK += [
        f"-L{libomp_prefix}/lib",
        "-lomp",
        "-Wl,-rpath," + f"{libomp_prefix}/lib",
    ]
else:
    # gcc / linux clang ship OpenMP via -fopenmp directly.
    EXTRA_COMPILE += ["-fopenmp"]
    EXTRA_LINK += ["-fopenmp"]


def make_ext(qualified_name: str, source: str) -> Extension:
    return Extension(
        name=qualified_name,
        sources=[source],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE,
        extra_link_args=EXTRA_LINK,
        language="c++",
    )


ext_modules = [
    # Extension names use the same dotted path the Python code imports them
    # under (`from src.parallel import cython_X`). pyproject.toml no longer
    # carries `package-dir = {"" = "src"}`, so setuptools won't double-apply
    # the prefix — the .so files land directly at `src/parallel/*.so`.
    make_ext("src.parallel.cython_encoder", "src/parallel/cython_encoder.pyx"),
    make_ext("src.parallel.cython_index",   "src/parallel/cython_index.pyx"),
    # Decoder-side differentiable top-k (Avner). Building it from this single
    # setup.py gives `python setup.py build_ext --inplace` one-shot semantics.
    make_ext("src.parallel.cython_topk",    "src/parallel/cython_topk.pyx"),
    make_ext("src.parallel.cython_loss",    "src/parallel/cython_loss.pyx"),
    make_ext("src.parallel.cython_decoder",    "src/parallel/cython_decoder.pyx"),
]

setup(
    name="clarion",
    version="0.1.0",
    description="Continuous Latent Augmented Retrieval Inference on N-cores",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        },
        annotate=False,
    ),
    zip_safe=False,
)
