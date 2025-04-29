#!/usr/bin/env python
import os
import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++17', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++17', '-O3'
  ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
    ext_modules=[
      setuptools.Extension(
        'kimimaro.skeletontricks',
        sources=[ './ext/skeletontricks/skeletontricks.pyx' ],
        language='c++',
        include_dirs=[ str(NumpyImport()) ],
        extra_compile_args=extra_compile_args,
      ),
    ],
)