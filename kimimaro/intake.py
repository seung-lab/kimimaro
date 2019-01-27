from collections import defaultdict

import numpy as np
import scipy.ndimage
from tqdm import tqdm

import cloudvolume
from cloudvolume import CloudVolume, PrecomputedSkeleton, Bbox

import cc3d # connected components
import edt # euclidean distance transform
import fastremap

import kimimaro.skeletontricks
import kimimaro.trace

class DimensionError(Exception):
  pass

def skeletonize(
    all_labels, teasar_params, anisotropy=(1,1,1),
    object_ids=None, dust_threshold=1000, cc_safety_factor=0.25,
    progress=False
  ):
  """

  """
  if all_labels.ndim not in (2,3):
    raise DimensionError("Can only skeletonize arrays of dimension 2 or 3.")

  if all_labels.ndim == 2:
    all_labels = all_labels[..., np.newaxis ]

  anisotropy = np.array(anisotropy, dtype=np.float32)

  all_labels = apply_object_mask(all_labels, object_ids)
  if not np.any(all_labels):
    return

  cc_labels, remapping = compute_cc_labels(all_labels, cc_safety_factor)
  del all_labels

  all_dbf = edt.edt(cc_labels, 
    anisotropy=anisotropy,
    black_border=False,
    order='F',
  )
  # slows things down, but saves memory
  # max_all_dbf = np.max(all_dbf)
  # if max_all_dbf < np.finfo(np.float16).max:
  #   all_dbf = all_dbf.astype(np.float16)

  cc_segids, pxct = np.unique(cc_labels, return_counts=True)
  cc_segids = [ sid for sid, ct in zip(cc_segids, pxct) if ct > dust_threshold ]

  all_slices = scipy.ndimage.find_objects(cc_labels)

  skeletons = defaultdict(list)
  for segid in tqdm(cc_segids, disable=(not progress), desc="Label"):
    if segid == 0:
      continue 

    # Crop DBF to ROI
    slices = all_slices[segid - 1]
    if slices is None:
      continue

    labels = cc_labels[slices]
    labels = (labels == segid)
    dbf = (labels * all_dbf[slices]).astype(np.float32)

    roi = Bbox.from_slices(slices)

    skeleton = kimimaro.trace.trace(labels, dbf, anisotropy=anisotropy, **teasar_params)
    skeleton.vertices[:,0] += roi.minpt.x
    skeleton.vertices[:,1] += roi.minpt.y
    skeleton.vertices[:,2] += roi.minpt.z

    if skeleton.empty():
      continue

    orig_segid = remapping[segid]
    skeleton.id = orig_segid
    skeleton.vertices *= anisotropy
    skeletons[orig_segid].append(skeleton)

  return merge(skeletons)

def apply_object_mask(all_labels, object_ids):
  if object_ids is None:
    return all_labels

  if len(object_ids) == 1:
    all_labels = kimimaro.skeletontricks.zero_out_all_except(all_labels, object_ids[0]) # faster
  else:
    all_labels[~np.isin(all_labels, object_ids)] = 0

  return all_labels

def compute_cc_labels(all_labels, cc_safety_factor):
  if cc_safety_factor <= 0 or cc_safety_factor > 1:
    raise ValueError("cc_safety_factor must be greater than zero and less than or equal to one. Got: " + str(cc_safety_factor))

  tmp_labels = all_labels
  if np.dtype(all_labels.dtype).itemsize > 1:
    tmp_labels, remapping = fastremap.renumber(all_labels)

  cc_labels = cc3d.connected_components(tmp_labels, max_labels=int(tmp_labels.size * cc_safety_factor))

  del tmp_labels
  remapping = kimimaro.skeletontricks.get_mapping(all_labels, cc_labels) 
  return cc_labels, remapping

def merge(skeletons):
  merged_skels = {}
  for segid, skels in skeletons.items():
    skel = PrecomputedSkeleton.simple_merge(skels)
    merged_skels[segid] = skel.consolidate()

  return merged_skels