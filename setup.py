#!/usr/bin/env python
import os
import setuptools

import numpy as np

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

# NOTE: If skeletontricks.cpp does not exist, you must run
# cython -3 --cplus ./ext/skeletontricks/skeletontricks.pyx

setuptools.setup(
  name="kimimaro",
  version="2.2.0",
  setup_requires=["numpy"],
  install_requires=[
    "click",
    "connected-components-3d>=1.5.0",
    "cloud-volume>=0.57.6",
    "dijkstra3d>=1.9.0",
    "fill-voids>=2.0.0",
    "edt>=2.1.0",
    "fastremap>=1.10.2",
    "networkx",
    "numpy>=1.16.1",
    "pathos",
    "pytest",
    "scipy>=1.1.0",
  ],
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
  author="William Silversmith, Alex Bae, Forrest Collman",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  description="Skeletonize densely labeled image volumes.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  keywords = "volumetric-data numpy teasar skeletonization centerline medial-axis-transform centerline-extraction computer-vision-alogithms connectomics image-processing biomedical-image-processing voxel",
  url = "https://github.com/seung-lab/kimimaro/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ],
  entry_points={
    "console_scripts": [
      "kimimaro=kimimaro_cli:main"
    ],
  },
)

