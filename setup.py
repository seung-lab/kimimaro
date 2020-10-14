#!/usr/bin/env python
import os
import setuptools

import numpy as np

# NOTE: If skeletontricks.cpp does not exist, you must run
# cython -3 --cplus ./ext/skeletontricks/skeletontricks.pyx

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  ext_modules=[
    setuptools.Extension(
      'kimimaro.skeletontricks',
      sources=[ './ext/skeletontricks/skeletontricks.cpp' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=[
        '-std=c++11', '-O3', '-ffast-math'
      ]
    ),
  ],
  long_description_content_type='text/markdown',
  pbr=True,
)

