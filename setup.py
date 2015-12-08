# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from distutils.core import setup
from distutils.extension import Extension

# Third-party
import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Get numpy path
numpy_base_path = os.path.split(np.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

# malloc
mac_incl_path = "/usr/include/malloc"

# Get gary path
import gary
gary_base_path = os.path.split(gary.__file__)[0]
gary_incl_path = os.path.join(gary_base_path, "integrate", "cyintegrators", "dopri")

extensions = []

mle = Extension("streammorphology.extern.*",
                ["streammorphology/extern/*.pyx",
                 os.path.join(gary_incl_path,"dop853.c")],
                include_dirs=[numpy_incl_path, gary_incl_path, mac_incl_path],
                extra_compile_args=['-std=c99'])
extensions.append(mle)

setup(
    name="StreamMorphology",
    version="0.1",
    author="Adrian M. Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="MIT",
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    packages=["streammorphology",
              "streammorphology.extern",
              "streammorphology.ensemble",
              "streammorphology.freqmap",
              "streammorphology.lyapunov"],
    scripts=[]
)
