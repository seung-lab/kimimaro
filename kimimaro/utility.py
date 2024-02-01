from typing import Dict, Union, List

import copy

import numpy as np
import scipy.ndimage
from tqdm import tqdm

from cloudvolume import Skeleton, Bbox
import kimimaro.skeletontricks

import cc3d
import fastremap
import xs3d

def extract_skeleton_from_binary_image(image):
  verts, edges = kimimaro.skeletontricks.extract_edges_from_binary_image(image)
  return Skeleton(verts, edges)

def compute_cc_labels(all_labels):
  tmp_labels = all_labels
  if np.dtype(all_labels.dtype).itemsize > 1:
    tmp_labels, remapping = fastremap.renumber(all_labels, in_place=False)

  cc_labels = cc3d.connected_components(tmp_labels)
  cc_labels = fastremap.refit(cc_labels)

  del tmp_labels
  remapping = kimimaro.skeletontricks.get_mapping(all_labels, cc_labels) 
  return cc_labels, remapping

def find_objects(labels):
  """  
  scipy.ndimage.find_objects performs about 7-8x faster on C 
  ordered arrays, so we just do it that way and convert
  the results if it's in F order.
  """
  if labels.flags['C_CONTIGUOUS']:
    return scipy.ndimage.find_objects(labels)
  else:
    all_slices = scipy.ndimage.find_objects(labels.T)
    return [ (slcs and slcs[::-1]) for slcs in all_slices ]    

def cross_sectional_area(
  all_labels:np.ndarray, 
  skeletons:Union[Dict[int,Skeleton],List[Skeleton],Skeleton],
  anisotropy:np.ndarray = np.array([1,1,1], dtype=np.float32),
  smoothing_window:int = 1,
  progress:bool = False,
) -> Union[Dict[int,Skeleton],List[Skeleton],Skeleton]:
  """
  Given a set of skeletons, find the cross sectional area
  for each vertex indicated by the sectioning plane
  defined by the vector pointing to the next vertex.

  When the smoothing_window is >1, these plane normal 
  vectors will be smoothed with a rolling average. This
  is useful since there can be high frequency
  oscillations in the skeleton.
  """
  prop = {
    "id": "cross_sectional_area",
    "data_type": "float32",
    "num_components": 1,
  }

  iterator = skeletons
  if type(skeletons) == dict:
    iterator = skeletons.values()
    total = len(skeletons)
  elif type(skeletons) == Skeleton:
    iterator = [ skeletons ]
    total = 1
  else:
    total = len(skeletons)

  cc_labels, remapping = compute_cc_labels(all_labels)
  remapping = { v:k for k,v in remapping.items() }

  all_slices = find_objects(cc_labels)
  del all_labels

  for skel in tqdm(iterator, desc="Labels", disable=(not progress), total=total):
    label = skel.id

    if label == 0:
      continue

    label = remapping[label]
    slices = all_slices[label - 1]
    if slices is None:
      continue

    roi = Bbox.from_slices(slices)
    if roi.volume() <= 1:
      continue

    binimg = np.asfortranarray(cc_labels[slices] == label)

    all_verts = (skel.vertices / anisotropy).round().astype(int)
    all_verts -= roi.minpt

    mapping = { tuple(v): i for i, v in enumerate(all_verts) }

    areas = np.zeros([all_verts.shape[0]], dtype=np.float32)

    paths = skel.paths()

    normal = np.array([1,0,0], dtype=np.float32)

    for path in paths:
      path = (path / anisotropy).round().astype(int)
      path -= roi.minpt

      normals = (path[1:] - path[:-1]).astype(np.float32)
      normals = np.concatenate([ normals, [normals[-1]] ])
      normals = moving_average(normals, smoothing_window)

      for i in range(len(normals)):
        normal = normals[i,:]
        normal /= np.linalg.norm(normal)        

      for i, vert in enumerate(path):
        idx = mapping[tuple(vert)]
        normal = normals[i]

        if areas[idx] == 0:
          areas[idx] = xs3d.cross_sectional_area(
            binimg, vert, 
            normal, anisotropy,
          )

    skel.extra_attributes.append(prop)
    skel.cross_sectional_area = areas

  return skeletons

# From SO: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average(a:np.ndarray, n:int) -> np.ndarray:
  if n <= 0:
    raise ValueError(f"Window size ({n}), must be >= 1.")
  elif n == 1:
    return a
  mirror = (len(a) - (len(a) - n + 1)) / 2
  extra = 0
  if mirror != int(mirror):
    extra = 1
  mirror = int(mirror)
  a = np.concatenate([ [a[0] ] * (mirror + extra), a, [ a[-1] ] * mirror ])
  ret = np.cumsum(a, dtype=float, axis=0)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

