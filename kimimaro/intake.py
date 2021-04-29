"""
This file is part of Kimimaro.

Kimimaro is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Kimimaro is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Kimimaro.  If not, see <https://www.gnu.org/licenses/>.
"""

from collections import defaultdict
from functools import partial
import gc
import multiprocessing as mp
import signal
import uuid

import numpy as np
import pathos.pools
import scipy.ndimage
import scipy.spatial
from tqdm import tqdm

import cloudvolume
from cloudvolume import CloudVolume, PrecomputedSkeleton, Bbox
import cloudvolume.sharedmemory as shm

import cc3d # connected components
import edt # euclidean distance transform
import fastremap
import fill_voids

import kimimaro.skeletontricks
import kimimaro.trace

class DimensionError(Exception):
  pass

DEFAULT_TEASAR_PARAMS = {
  'scale': 10, 'const': 50,
  'pdrf_scale': 100000,
  'pdrf_exponent': 4,
}

def skeletonize(
    all_labels, teasar_params=DEFAULT_TEASAR_PARAMS, anisotropy=(1,1,1),
    object_ids=None, dust_threshold=1000, cc_safety_factor=1,
    progress=False, fix_branching=True, in_place=False, 
    fix_borders=True, parallel=1, parallel_chunk_size=100,
    extra_targets_before=[], extra_targets_after=[],
    fill_holes=False, fix_avocados=False,
    voxel_graph=None
  ):
  """
  Skeletonize all non-zero labels in a given 2D or 3D image.

  Required:
    all_labels: a 2D or 3D numpy array of integer type (signed or unsigned) 

  Optional:
    anisotropy: the physical dimensions of each axis (e.g. 4nm x 4nm x 40nm)
    object_ids: If not none, zero out all labels other than those specified here.
    teasar_params: {
      scale: during the "rolling ball" invalidation phase, multiply 
          the DBF value by this.
      const: during the "rolling ball" invalidation phase, this 
          is the minimum radius in chosen physical units (i.e. nm).
      soma_detection_threshold: if object has a DBF value larger than this, 
          root will be placed at largest DBF value and special one time invalidation
          will be run over that root location (see soma_invalidation scale)
          expressed in chosen physical units (i.e. nm) 
      pdrf_scale: scale factor in front of dbf, used to weight dbf over euclidean distance (higher to pay more attention to dbf) (default 5000)
      pdrf_exponent: exponent in dbf formula on distance from edge, faster if factor of 2 (default 16)
      soma_invalidation_scale: the 'scale' factor used in the one time soma root invalidation (default .5)
      soma_invalidation_const: the 'const' factor used in the one time soma root invalidation (default 0)
                             (units in chosen physical units (i.e. nm))
      max_paths: max paths to trace on a single object. Moves onto the next object after this point.
    }
    dust_threshold: don't bother skeletonizing connected components smaller than
      this many voxels.
    fill_holes: preemptively run a void filling algorithm on all connected
      components and delete labels that get filled in. This can improve the
      quality of the reconstruction if holes in the shapes are artifacts introduced
      by the segmentation pipeline. This option incurs moderate overhead.

      WARNING: THIS WILL REMOVE INPUT LABELS THAT ARE DEEMED TO BE HOLES.

    cc_safety_factor: Value between 0 and 1 that scales the size of the 
      disjoint set maps in connected_components. 1 is guaranteed to work,
      but is probably excessive and corresponds to every pixel being a different
      label. Use smaller values to save some memory.

    extra_targets_before: List of x,y,z voxel coordinates that will all 
      be traced to from the root regardless of whether those points have 
      been invalidated. These targets will be applied BEFORE the regular
      target selection algorithm is run.      

      e.g. [ (x,y,z), (x,y,z) ]

    extra_targets_after: Same as extra_targets_before but the additional
      targets will be applied AFTER the usual algorithm runs.

    progress: if true, display a progress bar
    fix_branching: When enabled, zero the edge weights by of previously 
      traced paths. This causes branch points to occur closer to 
      the actual path divergence. However, there is a performance penalty
      associated with this as dijkstra's algorithm is computed once per a path
      rather than once per a skeleton.
    in_place: if true, allow input labels to be modified to reduce
      memory usage and possibly improve performance.
    fix_borders: ensure that segments touching the border place a 
      skeleton endpoint in a predictable place to make merging 
      adjacent chunks easier.
    fix_avocados: If nuclei are segmented seperately from somata
      then we can try to detect and fix this issue.
    voxel_graph: a connection graph that defines permissible 
      directions of motion between voxels. This is useful for
      dealing with self-touches. The graph is defined by the
      conventions used in cc3d.voxel_connectivity_graph 
      (https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92)
    parallel: number of subprocesses to use.
      <= 0: Use multiprocessing.count_cpu() 
         1: Only use the main process.
      >= 2: Use this number of subprocesses.
    parallel_chunk_size: default number of skeletons to 
      submit to each parallel process before returning results,
      updating the progress bar, and submitting a new task set. 
      Setting this number too low results in excess IPC overhead,
      and setting it too high can result in task starvation towards
      the end of a job and infrequent progress bar updates. If the
      chunk size is set higher than num tasks // parallel, that number
      is used instead.

  Returns: { $segid: cloudvolume.PrecomputedSkeleton, ... }
  """

  anisotropy = np.array(anisotropy, dtype=np.float32)

  all_labels = format_labels(all_labels, in_place=in_place)
  all_labels = apply_object_mask(all_labels, object_ids)

  if all_labels.size <= dust_threshold:
    return {}
  
  minlabel, maxlabel = fastremap.minmax(all_labels)

  if minlabel == 0 and maxlabel == 0:
    return {}

  cc_labels, remapping = compute_cc_labels(all_labels, cc_safety_factor)
  del all_labels

  if fill_holes:
    cc_labels = fill_all_holes(cc_labels, progress)

  extra_targets_before = points_to_labels(extra_targets_before, cc_labels)
  extra_targets_after = points_to_labels(extra_targets_after, cc_labels)

  def edtfn(labels):
    return edt.edt(labels, 
      anisotropy=anisotropy,
      black_border=(minlabel == maxlabel),
      order='F',
      parallel=parallel,
      voxel_graph=voxel_graph,
    )

  all_dbf = edtfn(cc_labels)
 
  if fix_avocados:
    cc_labels, all_dbf, remapping = engage_avocado_protection(
      cc_labels, all_dbf, remapping,
      soma_detection_threshold=teasar_params.get('soma_detection_threshold', 0),
      edtfn=edtfn,
      progress=progress,
    )

  cc_segids, pxct = fastremap.unique(cc_labels, return_counts=True)
  cc_segids = [ sid for sid, ct in zip(cc_segids, pxct) if ct > dust_threshold and sid != 0 ]

  all_slices = find_objects(cc_labels)

  border_targets = defaultdict(list)
  if fix_borders:
    border_targets = compute_border_targets(cc_labels, anisotropy)

  print_quotes(parallel) # easter egg

  if parallel <= 0:
    parallel = mp.cpu_count()

  if parallel == 1:
    return skeletonize_subset(
      all_dbf, cc_labels, voxel_graph, remapping, 
      teasar_params, anisotropy, all_slices, 
      border_targets, extra_targets_before, extra_targets_after,
      progress, fix_borders, fix_branching, 
      cc_segids
    )
  else:
    # The following section can't be moved into 
    # skeletonize parallel because then all_dbf 
    # and cc_labels can't be deleted to save memory.
    suffix = uuid.uuid1().hex

    dbf_shm_location = 'kimimaro-shm-dbf-' + suffix
    cc_shm_location = 'kimimaro-shm-cc-labels-' + suffix
    vg_shm_location = 'kimimaro-shm-voxel-graph-' + suffix

    dbf_mmap, all_dbf_shm = shm.ndarray( all_dbf.shape, all_dbf.dtype, dbf_shm_location, order='F')
    all_dbf_shm[:] = all_dbf 
    del all_dbf 

    cc_mmap, cc_labels_shm = shm.ndarray( cc_labels.shape, cc_labels.dtype, cc_shm_location, order='F')    
    cc_labels_shm[:] = cc_labels 
    del cc_labels

    voxel_graph_shm = None
    vg_mmap = None
    if voxel_graph is not None:
      vg_mmap, voxel_graph_shm = shm.ndarray( voxel_graph.shape, voxel_graph.dtype, vg_shm_location, order='F')    
      voxel_graph_shm[:] = voxel_graph
      del voxel_graph

    skeletons = skeletonize_parallel(      
      all_dbf_shm, dbf_shm_location, 
      cc_labels_shm, cc_shm_location, remapping, 
      voxel_graph_shm, vg_shm_location,
      teasar_params, anisotropy, all_slices, 
      border_targets, extra_targets_before, extra_targets_after,
      progress, fix_borders, fix_branching, 
      cc_segids, parallel, parallel_chunk_size
    )

    dbf_mmap.close()
    cc_mmap.close()
    if vg_mmap:
      vg_mmap.close()

    return skeletons

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

def format_labels(labels, in_place):
  if in_place:
    labels = fastremap.asfortranarray(labels)
  else:
    labels = np.copy(labels, order='F')

  if labels.dtype == bool:
    labels = labels.view(np.uint8)

  original_shape = labels.shape

  while labels.ndim < 3:
    labels = labels[..., np.newaxis ]

  while labels.ndim > 3:
    if labels.shape[-1] == 1:
      labels = labels[..., 0]
    else:
      raise DimensionError(
        "Input labels may be no more than three non-trivial dimensions. Got: {}".format(
          original_shape
        )
      )

  return labels

def skeletonize_parallel(
    all_dbf_shm, dbf_shm_location, 
    cc_labels_shm, cc_shm_location, remapping, 
    voxel_graph_shm, vg_shm_location,
    teasar_params, anisotropy, all_slices, 
    border_targets, extra_targets_before, extra_targets_after,
    progress, fix_borders, fix_branching, 
    cc_segids, parallel, chunk_size
  ):
    prevsigint = signal.getsignal(signal.SIGINT)
    prevsigterm = signal.getsignal(signal.SIGTERM)
    
    executor = pathos.pools.ProcessPool(parallel)

    def cleanup(signum, frame):
      shm.unlink(dbf_shm_location)
      shm.unlink(cc_shm_location)
      executor.terminate()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)   

    vg_shape = voxel_graph_shm.shape if voxel_graph_shm else None
    vg_dtype = voxel_graph_shm.dtype if voxel_graph_shm else None

    skeletonizefn = partial(parallel_skeletonize_subset, 
      dbf_shm_location, all_dbf_shm.shape, all_dbf_shm.dtype, 
      cc_shm_location, cc_labels_shm.shape, cc_labels_shm.dtype,
      vg_shm_location, vg_shape, vg_dtype,
      remapping, teasar_params, anisotropy, all_slices, 
      border_targets, extra_targets_before, extra_targets_after, 
      False, # progress, use our own progress bar below
      fix_borders, fix_branching, 
    )

    ccids = []
    if chunk_size < len(cc_segids) // parallel:
      for i in range(0, len(cc_segids), chunk_size):
        ccids.append(cc_segids[i:i+chunk_size])
    else:
      for i in range(parallel):
        ccids.append(cc_segids[i::parallel])

    skeletons = defaultdict(list)
    with tqdm(total=len(cc_segids), disable=(not progress), desc="Skeletonizing Labels") as pbar:
      for skels in executor.uimap(skeletonizefn, ccids):
        for segid, skel in skels.items():
          skeletons[segid].append(skel)
        pbar.update(len(skels))
    executor.close()
    executor.join()
    executor.clear()

    signal.signal(signal.SIGINT, prevsigint)
    signal.signal(signal.SIGTERM, prevsigterm)
    
    shm.unlink(dbf_shm_location)
    shm.unlink(cc_shm_location)
    shm.unlink(vg_shm_location)

    return merge(skeletons)

def parallel_skeletonize_subset(    
    dbf_shm_location, dbf_shape, dbf_dtype, 
    cc_shm_location, cc_shape, cc_dtype, 
    vg_shm_location, vg_shape, vg_dtype,
    *args, **kwargs
  ):
  
  dbf_mmap, all_dbf = shm.ndarray( dbf_shape, dtype=dbf_dtype, location=dbf_shm_location, order='F')
  cc_mmap, cc_labels = shm.ndarray( cc_shape, dtype=cc_dtype, location=cc_shm_location, order='F')

  if vg_shape is None:
    vg_mmap, voxel_graph = None, None
  else:
    vg_mmap, voxel_graph = shm.ndarray( vg_shape, dtype=vg_dtype, location=vg_shm_location, order='F')

  skels = skeletonize_subset(all_dbf, cc_labels, voxel_graph, *args, **kwargs)

  dbf_mmap.close()
  cc_mmap.close()
  if vg_mmap:
    vg_mmap.close()

  return skels

def skeletonize_subset(
    all_dbf, cc_labels, voxel_graph, remapping, 
    teasar_params, anisotropy, all_slices, 
    border_targets, extra_targets_before, extra_targets_after,
    progress, fix_borders, fix_branching, 
    cc_segids
  ):

  skeletons = defaultdict(list)
  for segid in tqdm(cc_segids, disable=(not progress), desc="Skeletonizing Labels"):
    # Crop DBF to ROI
    slices = all_slices[segid - 1]
    if slices is None:
      continue

    roi = Bbox.from_slices(slices)
    if roi.volume() <= 1:
      continue

    labels = cc_labels[slices]
    labels = (labels == segid)
    dbf = (labels * all_dbf[slices]).astype(np.float32)
    cropped_voxel_graph = (voxel_graph[slices] if voxel_graph is not None else None)

    manual_targets_before = []
    manual_targets_after = []
    root = None 

    def translate_to_roi(targets):
      targets = np.array(targets)
      targets -= roi.minpt.astype(np.uint32)
      return targets.tolist()      

    # We only source a predetermined root from 
    # border_targets because we understand that it's
    # located at a reasonable place at the edge of the
    # shape. In theory, extra targets can be positioned
    # anywhere within the shape or off the shape, making it 
    # a dicey proposition. 
    if len(border_targets[segid]) > 0:
      manual_targets_before = translate_to_roi(border_targets[segid])
      root = manual_targets_before.pop()

    if segid in extra_targets_before and len(extra_targets_before[segid]) > 0:
      manual_targets_before.extend( translate_to_roi(extra_targets_before[segid]) )

    if segid in extra_targets_after and len(extra_targets_after[segid]) > 0:
      manual_targets_after.extend( translate_to_roi(extra_targets_after[segid]) )

    skeleton = kimimaro.trace.trace(
      labels, 
      dbf, 
      anisotropy=anisotropy, 
      fix_branching=fix_branching, 
      manual_targets_before=manual_targets_before,
      manual_targets_after=manual_targets_after,
      root=root,
      voxel_graph=cropped_voxel_graph,
      **teasar_params
    )

    if skeleton.empty():
      continue

    skeleton.vertices += roi.minpt

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
    all_labels = fastremap.mask_except(all_labels, object_ids, in_place=True)

  return all_labels

def compute_cc_labels(all_labels, cc_safety_factor):
  if cc_safety_factor <= 0 or cc_safety_factor > 1:
    raise ValueError("cc_safety_factor must be greater than zero and less than or equal to one. Got: " + str(cc_safety_factor))

  tmp_labels = all_labels
  if np.dtype(all_labels.dtype).itemsize > 1:
    tmp_labels, remapping = fastremap.renumber(all_labels, in_place=False)

  cc_labels = cc3d.connected_components(tmp_labels, max_labels=int(tmp_labels.size * cc_safety_factor))
  cc_labels = fastremap.refit(cc_labels)

  del tmp_labels
  remapping = kimimaro.skeletontricks.get_mapping(all_labels, cc_labels) 
  return cc_labels, remapping

def points_to_labels(pts, cc_labels):
  mapping = defaultdict(list)
  for pt in pts:
    pt = tuple(pt)
    mapping[ cc_labels[pt] ].append(pt)
  return mapping

def compute_border_targets(cc_labels, anisotropy):
  sx, sy, sz = cc_labels.shape

  planes = (
    ( cc_labels[:,:,0], (0, 1), lambda x,y: (x, y, 0) ),     # top xy
    ( cc_labels[:,:,-1], (0, 1), lambda x,y: (x, y, sz-1) ), # bottom xy
    ( cc_labels[:,0,:], (0, 2), lambda x,z: (x, 0, z) ),     # left xz
    ( cc_labels[:,-1,:], (0, 2), lambda x,z: (x, sy-1, z) ), # right xz
    ( cc_labels[0,:,:], (1, 2), lambda y,z: (0, y, z) ),     # front yz
    ( cc_labels[-1,:,:], (1, 2), lambda y,z: (sx-1, y, z) )  # back yz
  )

  target_list = defaultdict(set)

  for plane, dims, rotatefn in planes:
    wx, wy = anisotropy[dims[0]], anisotropy[dims[1]]
    plane = np.copy(plane, order='F')
    cc_plane = cc3d.connected_components(np.ascontiguousarray(plane))
    dt_plane = edt.edt(cc_plane, black_border=True, anisotropy=(wx, wy))

    plane_targets = kimimaro.skeletontricks.find_border_targets(
      dt_plane, cc_plane, wx, wy
    )

    plane = plane[..., np.newaxis]
    cc_plane = cc_plane[..., np.newaxis]
    remapping = kimimaro.skeletontricks.get_mapping(plane, cc_plane)

    for label, pt in plane_targets.items():
      label = remapping[label]
      target_list[label].add(
        rotatefn( int(pt[0]), int(pt[1]) )
      )

  target_list.default_factory = lambda: np.array([], np.uint32)
  for label, pts in target_list.items():
    target_list[label] = np.array(list(pts), dtype=np.uint32)

  return target_list

def merge(skeletons):
  merged_skels = {}
  for segid, skels in skeletons.items():
    skel = PrecomputedSkeleton.simple_merge(skels)
    merged_skels[segid] = skel.consolidate()

  return merged_skels

def argmax(arr):
  if arr.flags['C_CONTIGUOUS']:
    return np.unravel_index(np.argmax(arr), arr.shape, order='C')
  return np.unravel_index(np.argmax(arr.T), arr.shape, order='F')

def engage_avocado_protection(
  cc_labels, all_dbf, remapping,
  soma_detection_threshold, edtfn, 
  progress
):
  orig_cc_labels = np.copy(cc_labels, order='F')

  unchanged = set()
  max_iterations = max(fastremap.unique(cc_labels))

  # This loop handles nested avocados
  # Unless there are deeply nested double avocados,
  # this should complete in 2-3 passes. We limit it
  # to 20 just to make sure this loop terminates no matter what.
  # Avocados aren't the end of the world.
  for _ in tqdm(range(20), disable=(not progress), desc="Avocado Pass"): 
    # Note: Divide soma_detection_threshold by a bit more than 2 because the nucleii are going to be
    # about a factor of 2 or less smaller than what we'd expect from a cell. For example,
    # in an avocado I saw, the DBF of the nucleus was 499 when the detection threshold was 
    # set to 1100.
    candidates = set(fastremap.unique(cc_labels * (all_dbf > soma_detection_threshold / 2.5)))
    candidates -= unchanged
    candidates.discard(0)

    cc_labels, unchanged_this_cycle, changes = engage_avocado_protection_single_pass(
      cc_labels, all_dbf,
      candidates=candidates,
      progress=progress,
    )
    unchanged |= unchanged_this_cycle

    if len(changes) == 0:
      break 
    
    all_dbf = edtfn(cc_labels)

  # Downstream logic assumes cc_labels is contigiously numbered
  cc_labels, _ = fastremap.renumber(cc_labels, in_place=True)
  cc_remapping = kimimaro.skeletontricks.get_mapping(orig_cc_labels, cc_labels)

  adjusted_remapping = {}
  for new_cc, cc in cc_remapping.items():
    if cc in remapping:
      adjusted_remapping[new_cc] = remapping[cc]

  return cc_labels, all_dbf, adjusted_remapping

def engage_avocado_protection_single_pass(
  cc_labels, all_dbf, 
  candidates=None, progress=False
):
  """
  For each candidate, check if there's a fruit around the
  avocado pit roughly from the center (the max EDT).
  """

  if candidates is None:
    candidates = fastremap.unique(cc_labels)

  candidates = [ label for label in candidates if label != 0 ]

  unchanged = set()
  changed = set()

  if len(candidates) == 0:
    return cc_labels, unchanged, changed

  def paint_walls(binimg):
    """
    Ensure that inclusions that touch the wall are handled
    by performing a 2D fill on each wall.
    """
    binimg[:,:,0 ] = fill_voids.fill(binimg[:,:,0 ])
    binimg[:,:,-1] = fill_voids.fill(binimg[:,:,-1])
    binimg[:,0,: ] = fill_voids.fill(binimg[:,0,: ])
    binimg[:,-1,:] = fill_voids.fill(binimg[:,-1,:])
    binimg[0,:,: ] = fill_voids.fill(binimg[0,:,: ])
    binimg[-1,:,:] = fill_voids.fill(binimg[-1,:,:])
    return binimg

  remap = {}
  for label in tqdm(candidates, disable=(not progress), desc="Fixing Avocados"):
    binimg = paint_walls(cc_labels == label) # image of the pit
    coord = argmax(binimg * all_dbf)

    (pit, fruit) = kimimaro.skeletontricks.find_avocado_fruit(
      cc_labels, coord[0], coord[1], coord[2]
    )
    if pit == fruit and pit not in changed:
      unchanged.add(pit)
    else:
      unchanged.discard(pit)
      unchanged.discard(fruit)
      changed.add(pit)
      changed.add(fruit)
      binimg |= (cc_labels == fruit)
    
    binimg, N = fill_voids.fill(binimg, in_place=True, return_fill_count=True)
    cc_labels *= ~binimg
    cc_labels += label * binimg

  return cc_labels, unchanged, changed

def synapses_to_targets(labels, synapses, progress=False):
  """
  Turn the output of synapse detection and assignment, usually 
  centroid + pre/post into actionable targets. For a given 
  labeled volume, take the centroid and a pre or post label
  and find the nearest voxel for that label and add the coordinates
  of that voxel to a list of targets.

  labels: a 3d array containing labels
  synapses: { label: [ (centroid, swc_label), (centroid, swc_label), ... ] }
    where centroid is an (x,y,z) float triple in voxel coordinate space
      where the origin is the same as for labels
    where swc_label is the label to be added to the vertex attributes for
      the resulting target.
    where label is a presynaptic OR a postsynaptic label
      (submit two items to cover both)

  Returns: { (x,y,z): swc_label, ... } targets for skeletonization
  """
  while labels.ndim > 3:
    labels = labels[...,0]

  targets = {}

  for label, pairs in tqdm(synapses.items(), disable=(not progress), desc='Converting Synapses to Targets'):
    point_cloud = np.vstack((labels == label).nonzero()).T # [ [x,y,z], ... ]
    if len(point_cloud) == 0:
      continue

    swc_labels = defaultdict(list) 
    for centroid, swc_label in pairs:
      swc_labels[swc_label].append(centroid)

    for swc_label, centroids in swc_labels.items():
      distances = scipy.spatial.distance.cdist(point_cloud, centroids)
      minima = np.unique(np.argmin(distances, axis=0))
      tmp_targets = [ tuple(point_cloud[idx]) for idx in minima ]
      targets.update({ target: swc_label for target in tmp_targets })

  return targets

def fill_all_holes(cc_labels, progress=False, return_fill_count=False):
  """
  Fills the holes in each connected component and removes components that
  get filled in. The idea is that holes (entirely contained labels or background) 
  are artifacts in cell segmentations. A common example is a nucleus segmented 
  separately from the rest of the cell or errors in a manual segmentation leaving
  a void in a dendrite.

  cc_labels: an image containing connected components with labels smaller than
    the number of voxels in the image.
  progress: Display a progress bar or not.
  return_fill_count: if specified, return a tuple (filled_image, N) where N is
    the number of voxels that were filled in.

  Returns: filled_in_labels
  """
  labels = fastremap.unique(cc_labels)
  labels_set = set(labels)
  labels_set.discard(0)

  all_slices = find_objects(cc_labels)
  pixels_filled = 0

  for label in tqdm(labels, disable=(not progress), desc="Filling Holes"):
    if label not in labels_set:
      continue

    slices = all_slices[label - 1]
    if slices is None:
      continue

    binary_image = (cc_labels[slices] == label)
    binary_image, N = fill_voids.fill(
      binary_image, in_place=True, 
      return_fill_count=True
    )
    pixels_filled += N
    if N == 0:
      continue 

    sub_labels = set(fastremap.unique(cc_labels[slices] * binary_image))
    sub_labels.remove(label)
    labels_set -= sub_labels
    cc_labels[slices] = cc_labels[slices] * ~binary_image + label * binary_image

  if return_fill_count:
    return cc_labels, pixels_filled
  return cc_labels

def print_quotes(parallel):
  if parallel == -1:
    print("Against the power of will I possess... The capability of my body is nothing.")
  elif parallel == -2:
    print("I will see the truth of this world... OROCHIMARU-SAMA WILL SHOW ME!!!")

  if -2 <= parallel < 0:
    print("CURSED SEAL OF THE EARTH!!!")  
