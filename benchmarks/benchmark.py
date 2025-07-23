import time
import numpy as np
import kimimaro
import crackle
import pickle

labels = crackle.load("connectomics.npy.ckl.gz")

s = time.time()
skels = kimimaro.skeletonize(
  labels, 
  teasar_params={
    'scale': 1.5,
    'const': 300, # physical units
    'pdrf_exponent': 4,
    'pdrf_scale': 100000,
    'soma_detection_threshold': 1100, # physical units
    'soma_acceptance_threshold': 3500, # physical units
    'soma_invalidation_scale': 1.0,
    'soma_invalidation_const': 300, # physical units
    # 'max_paths': 50, # default None
  },
  # object_ids=[ ], # process only the specified labels
  # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
  # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
  # dust_threshold=1000, # skip connected components with fewer than this many voxels
  anisotropy=(16,16,40), # default True
  # fix_branching=True, # default True
  # fix_borders=True, # default True
  # fill_holes=False, # default False
  # fix_avocados=False, # default False
  progress=True, # default False, show progress bar
  # parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
  # parallel_chunk_size=100, # how many skeletons to process before updating progress bar
)
print(time.time() - s)

# with open("skels.pkl", "wb") as f:
#   pickle.dump(skels, f)

# with open("skels.pkl", "rb") as f:
#   skels = pickle.load(f)

s = time.time()
skels = kimimaro.utils.cross_sectional_area(
  labels, skels,
  anisotropy=(16,16,40),
  smoothing_window=7,
  progress=True,
  step=1,
)
print(f"{time.time() - s:.3f}s")