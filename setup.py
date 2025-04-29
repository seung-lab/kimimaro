#!/usr/bin/env python
import os
import setuptools

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

setuptools.setup(
    ext_modules=[
      setuptools.Extension(
        'kimimaro.skeletontricks',
        sources=[ './ext/skeletontricks/skeletontricks.pyx' ],
        language='c++',
        include_dirs=[ str(NumpyImport()) ],
        extra_compile_args=[
          '-std=c++17', '-O3', '-ffast-math'
        ]
      ),
    ],
)