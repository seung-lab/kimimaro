from typing import Dict

import copy

import numpy as np
import scipy.ndimage

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

def compute_cross_sectional_area(
  all_labels:np.ndarray, 
  skeletons:Dict[int,Skeleton],
  resolution:np.ndarray,
) -> Dict[int,Skeleton]:

  from tqdm import tqdm

  uniq = fastremap.unique(all_labels)

  prop = {
    "id": "cross_sectional_areas",
    "data_type": "float32",
    "num_components": 1,
  }


  all_slices = find_objects(all_labels)

  for label, skel in tqdm(skeletons.items()):
    if label == 0:
      continue

    slices = all_slices[label - 1]
    if slices is None:
      continue

    roi = Bbox.from_slices(slices)
    if roi.volume() <= 1:
      continue

    binimg = np.asfortranarray(all_labels[slices] == label)

    all_verts = (skel.vertices / resolution).round().astype(int)
    all_verts -= roi.minpt

    mapping = { tuple(v): i for i, v in enumerate(all_verts) }

    areas = np.zeros([all_verts.shape[0]], dtype=np.float32)

    paths = skel.paths()

    normal = np.array([1,0,0], dtype=np.float32)

    for path in paths:
      path = (path / resolution).round().astype(int)
      path -= roi.minpt

      prev = None
      if len(path):
        prev = np.copy(path[0])
        prev[0] -= 1

      for vert in tqdm(path):
        idx = mapping[tuple(vert)]
        normal = (vert - prev).astype(np.float32)
        normal /= np.linalg.norm(normal)

        if areas[idx] == 0:
          areas[idx] = xs3d.cross_sectional_area(
            binimg, vert, 
            normal, resolution,
          )

        prev = vert

    skel.extra_attributes.append(prop)
    skel.cross_sectional_areas = areas

  return skeletons











  
