from typing import Dict, Union, List, Tuple

import copy

import numpy as np
import scipy.ndimage
from tqdm import tqdm

from cloudvolume import Skeleton, Bbox, Vec
import kimimaro.skeletontricks

import cc3d
import dijkstra3d
import fastremap
import fill_voids
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

def add_property(skel, prop):
  needs_prop = True
  for skel_prop in skel.extra_attributes:
    if skel_prop["id"] == prop["id"]:
      needs_prop = False
      break

  if needs_prop:
    skel.extra_attributes.append(prop)

def shape_iterator(all_labels, skeletons, fill_holes, in_place, progress, fn):
  iterator = skeletons
  if type(skeletons) == dict:
    iterator = skeletons.values()
    total = len(skeletons)
  elif type(skeletons) == Skeleton:
    iterator = [ skeletons ]
    total = 1
  else:
    total = len(skeletons)

  if all_labels.dtype == bool:
    remapping = { True: 1, False: 0, 1:1, 0:0 }
  else:
    all_labels, remapping = fastremap.renumber(all_labels, in_place=in_place)

  all_slices = find_objects(all_labels)

  for skel in tqdm(iterator, desc="Labels", disable=(not progress), total=total):
    if all_labels.dtype == bool:
      label = 1
    else:
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

    roi.grow(1)
    roi.minpt = Vec.clamp(roi.minpt, Vec(0,0,0), roi.maxpt)
    slices = roi.to_slices()

    binimg = np.asfortranarray(all_labels[slices] == label)
    if fill_holes:
      binimg = fill_voids.fill(binimg, in_place=True)

    fn(skel, binimg, roi)

  return iterator

def cross_sectional_area(
  all_labels:np.ndarray, 
  skeletons:Union[Dict[int,Skeleton],List[Skeleton],Skeleton],
  anisotropy:np.ndarray = np.array([1,1,1], dtype=np.float32),
  smoothing_window:int = 1,
  progress:bool = False,
  in_place:bool = False,
  fill_holes:bool = False,
  repair_contacts:bool = False,
  visualize_section_planes:bool = False,
) -> Union[Dict[int,Skeleton],List[Skeleton],Skeleton]:
  """
  Given a set of skeletons, find the cross sectional area
  for each vertex indicated by the sectioning plane
  defined by the vector pointing to the next vertex.

  When the smoothing_window is >1, these plane normal 
  vectors will be smoothed with a rolling average. This
  is useful since there can be high frequency
  oscillations in the skeleton.

  This function will add the following attributes to
  each skeleton provided.

  skel.cross_sectional_area: float32 array of cross 
    sectional area per a vertex.

  skel.cross_sectional_area_contacts: uint8 array
    where non-zero entries indicate that the image
    border was contacted during the cross section
    computation, indicating a possible underestimate.

    The first six bits are a bitfield xxyyzz that
    tell you which image faces were touched and
    alternate from low (0) to high (size-1).

  repair_contacts: When True, only examine vertices
    that have a nonzero value for 
    skel.cross_sectional_area_contacts. This is intended
    to be used as a second pass after widening the image.

  visualize_section_planes: For debugging, paint section planes
    and display them using microviewer.
  """
  prop = {
    "id": "cross_sectional_area",
    "data_type": "float32",
    "num_components": 1,
  }

  def cross_sectional_area_helper(skel, binimg, roi):
    cross_sections = None
    if visualize_section_planes:
      cross_sections = np.zeros(binimg.shape, dtype=np.uint32, order="F")

    all_verts = (skel.vertices / anisotropy).round().astype(int)
    all_verts -= roi.minpt

    mapping = { tuple(v): i for i, v in enumerate(all_verts) }

    if repair_contacts:
      areas = skel.cross_sectional_area
      contacts = skel.cross_sectional_area_contacts
    else:
      areas = np.zeros([all_verts.shape[0]], dtype=np.float32)
      contacts = np.zeros([all_verts.shape[0]], dtype=np.uint8)

    paths = skel.paths()

    normal = np.array([1,0,0], dtype=np.float32)

    shape = np.array(binimg.shape)

    for path in paths:
      path = (path / anisotropy).round().astype(int)
      path -= roi.minpt

      normals = (path[1:] - path[:-1]).astype(np.float32)
      normals = np.concatenate([ normals, [normals[-1]] ])

      # Running the filter in the forward and then backwards
      # direction eliminates phase shift.
      normals = moving_average(normals, smoothing_window)
      normals = moving_average(normals[::-1], smoothing_window)[::-1]

      for i in range(len(normals)):
        normal = normals[i,:]
        normal /= np.linalg.norm(normal)        

      for i, vert in enumerate(path):
        if np.any(vert < 0) or np.any(vert > shape):
          continue

        idx = mapping[tuple(vert)]
        normal = normals[i]

        if areas[idx] == 0 or (repair_contacts and contacts[idx] > 0):
          areas[idx], contacts[idx] = xs3d.cross_sectional_area(
            binimg, vert, 
            normal, anisotropy,
            return_contact=True,
          )
          if visualize_section_planes:
            img = xs3d.cross_section(
              binimg, vert, 
              normal, anisotropy,
            )
            cross_sections[img > 0] = idx

    if visualize_section_planes:
      import microviewer
      microviewer.view(cross_sections, seg=True)

    add_property(skel, prop)

    skel.cross_sectional_area = areas
    skel.cross_sectional_area_contacts = contacts

  shape_iterator(
    all_labels, skeletons, 
    fill_holes, in_place, progress, 
    cross_sectional_area_helper
  )

  return skeletons

def oversegment(
  all_labels:np.ndarray, 
  skeletons:Union[Dict[int,Skeleton],List[Skeleton],Skeleton],
  anisotropy:np.ndarray = np.array([1,1,1], dtype=np.float32),
  progress:bool = False,
  fill_holes:bool = False,
  in_place:bool = False,
  downsample:int = 0,
) -> Tuple[np.ndarray, Union[Dict[int,Skeleton],List[Skeleton],Skeleton]]:
  """
  Use skeletons to create an oversegmentation of a pre-existing set
  of labels. This is useful for proofreading systems that work by merging
  labels.

  For each skeleton, get the feature map from its euclidean distance
  field. The final image is the composite of all these feature maps
  numbered from 1.

  Each skeleton will have a new property skel.segments that associates
  a label to each vertex.
  """
  prop = {
    "id": "segments",
    "data_type": "uint64",
    "num_components": 1,
  }

  skeletons = copy.deepcopy(skeletons)

  all_features = np.zeros(all_labels.shape, dtype=np.uint64, order="F")
  next_label = 0

  def oversegment_helper(skel, binimg, roi):
    nonlocal next_label
    nonlocal all_features

    segment_skel = skel
    if downsample > 0:
      segment_skel = skel.downsample(downsample)

    vertices = (segment_skel.vertices / anisotropy).round().astype(int)
    vertices -= roi.minpt

    field, feature_map = dijkstra3d.euclidean_distance_field(
      binimg, vertices, 
      anisotropy=anisotropy, 
      return_feature_map=True
    )
    del field

    add_property(skel, prop)

    vertices = (skel.vertices / anisotropy).round().astype(int)
    vertices -= roi.minpt

    feature_map[binimg] += next_label
    skel.segments = feature_map[vertices[:,0], vertices[:,1], vertices[:,2]]
    next_label += vertices.shape[0]
    all_features[roi.to_slices()] += feature_map

  # iterator is an iterable list of skeletons, not the shape iterator
  iterator = shape_iterator(
    all_labels, skeletons, fill_holes, in_place, progress, 
    oversegment_helper
  )

  all_features, mapping = fastremap.renumber(all_features)
  for skel in iterator:
    skel.segments = fastremap.remap(skel.segments, mapping, in_place=True)

  return all_features, skeletons

# From SO: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average(a:np.ndarray, n:int, mode:str = "symmetric") -> np.ndarray:
  if n <= 0:
    raise ValueError(f"Window size ({n}), must be >= 1.")
  elif n == 1:
    return a

  if len(a) == 0:
    return a

  if a.ndim == 2:
    a = np.pad(a, [[n, n],[0,0]], mode=mode)
  else:
    a = np.pad(a, [n, n], mode=mode)

  ret = np.cumsum(a, dtype=float, axis=0)
  ret = (ret[n:] - ret[:-n])[:-n]
  ret /= float(n)
  return ret

