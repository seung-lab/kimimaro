#!/usr/bin/env python
from __future__ import print_function
from distutils.command.build import build
from subprocess import call
import os
import shutil
import setuptools

import numpy as np

# NOTE: If skeletontricks.cpp does not exist, you must run
# cython -3 --cplus ./ext/skeletontricks/skeletontricks.pyx

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
     ':python_version == "2.7"': ['futures'],
  },
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

