"""
Skeletonization algorithm based on TEASAR (Sato et al. 2000).

Authors: Alex Bae and Will Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institue
Date: June 2018 - Februrary 2025

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
from math import log

import dijkstra3d
import edt
import fill_voids
import numpy as np
from scipy import ndimage

import kimimaro.skeletontricks

from osteoid import Skeleton

def trace(
    labels, DBF, 
    scale=10, const=10, anisotropy=(1,1,1), 
    soma_detection_threshold=1100, 
    soma_acceptance_threshold=4000, 
    pdrf_scale=5000, pdrf_exponent=16,
    soma_invalidation_scale=0.5,
    soma_invalidation_const=0,
    fix_branching=True,
    manual_targets_before=[],
    manual_targets_after=[],
    root=None,
    max_paths=None,
    voxel_graph=None,
  ):
  """
  Given the euclidean distance transform of a label ("Distance to Boundary Function"), 
  convert it into a skeleton using an algorithm based on TEASAR. 

  DBF: Result of the euclidean distance transform. Must represent a single label,
       assumed to be expressed in chosen physical units (i.e. nm)
  scale: during the "rolling ball" invalidation phase, multiply the DBF value by this.
  const: during the "rolling ball" invalidation phase, this is the minimum radius in chosen physical units (i.e. nm).
  anisotropy: (x,y,z) conversion factor for voxels to chosen physical units (i.e. nm)
  soma_detection_threshold: if object has a DBF value larger than this, 
    root will be placed at largest DBF value and special one time invalidation
    will be run over that root location (see soma_invalidation scale)
    expressed in chosen physical units (i.e. nm) 
  pdrf_scale: scale factor in front of dbf, used to weight dbf over euclidean distance (higher to pay more attention to dbf) (default 5000)
  pdrf_exponent: exponent in dbf formula on distance from edge, faster if factor of 2 (default 16)
  soma_invalidation_scale: the 'scale' factor used in the one time soma root invalidation (default .5)
  soma_invalidation_const: the 'const' factor used in the one time soma root invalidation (default 0)
                           (units in chosen physical units (i.e. nm))
  fix_branching: When enabled, zero out the graph edge weights traversed by 
    of previously found paths. This causes branch points to occur closer to 
    the actual path divergence. However, there is a large performance penalty
    associated with this as dijkstra's algorithm is computed once per a path
    rather than once per a skeleton.
  manual_targets_before: list of (x,y,z) that correspond to locations that must 
    have paths drawn to. Used for specifying root and border targets for
    merging adjacent chunks out-of-core. Targets are applied before ordinary
    target selection.
  manual_targets_after: Same as manual_targets_before but the additional 
    targets are applied after the usual algorithm runs. The current 
    invalidation status of the shape makes no difference.
  max_paths: If a label requires drawing this number of paths or more,
    abort and move onto the next label.
  root: If you want to force the root to be a particular voxel, you can
    specify it here.
  voxel_graph: a connection graph that defines permissible 
    directions of motion between voxels. This is useful for
    dealing with self-touches. The graph is defined by the
    conventions used in cc3d.voxel_connectivity_graph 
    (https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92)

  Based on the algorithm by:

  M. Sato, I. Bitter, M. Bender, A. Kaufman, and M. Nakajima. 
  "TEASAR: tree-structure extraction algorithm for accurate and robust skeletons"  
    Proc. the Eighth Pacific Conference on Computer Graphics and Applications. Oct. 2000.
    doi:10.1109/PCCGA.2000.883951 (https://ieeexplore.ieee.org/document/883951/)

  Returns: Skeleton object
  """
  dbf_max = np.max(DBF)
  labels = np.asfortranarray(labels)
  DBF = np.asfortranarray(DBF)

  soma_mode = False
  # > 5000 nm, gonna be a soma or blood vessel
  # For somata: specially handle the root by 
  # placing it at the approximate center of the soma
  if dbf_max > soma_detection_threshold:
    labels, num_voxels_filled = fill_voids.fill(labels, in_place=True, return_fill_count=True)
    if num_voxels_filled > 0:
      del DBF
      DBF = edt.edt(
        labels, 
        anisotropy=anisotropy, 
        black_border=np.all(labels),
        voxel_graph=voxel_graph,
      )
    dbf_max = np.max(DBF) 
    soma_mode = dbf_max > soma_acceptance_threshold

  soma_radius = 0.0

  if soma_mode:
    if root is not None:
      manual_targets_before.insert(0, root)
    root = find_soma_root(DBF, dbf_max)    
    soma_radius = dbf_max * soma_invalidation_scale + soma_invalidation_const
  elif root is None:
    root = find_root(labels, anisotropy, voxel_graph)
  
  if root is None:
    return Skeleton()
 
  free_space_radius = 0 if not soma_mode else DBF[root]
  # DBF: Distance to Boundary Field
  # DAF: Distance from any voxel Field (distance from root field)
  # PDRF: Penalized Distance from Root Field
  DBF = kimimaro.skeletontricks.zero2inf(DBF) # DBF[ DBF == 0 ] = np.inf
  DAF, target = dijkstra3d.euclidean_distance_field(
    labels, root, 
    anisotropy=anisotropy, 
    free_space_radius=free_space_radius,
    voxel_graph=voxel_graph,
    return_max_location=True,
  )
  DAF = kimimaro.skeletontricks.inf2zero(DAF) # DAF[ DAF == np.inf ] = 0
  PDRF = compute_pdrf(dbf_max, pdrf_scale, pdrf_exponent, DBF, DAF)

  # Use dijkstra propogation w/o a target to generate a field of
  # pointers from each voxel to its parent. Then we can rapidly
  # compute multiple paths by simply hopping pointers using path_from_parents
  if not fix_branching:
    parents = dijkstra3d.parental_field(PDRF, root, voxel_graph=voxel_graph)
    del PDRF
  else:
    parents = PDRF

  if soma_mode:
    invalidated, labels = kimimaro.skeletontricks.roll_invalidation_ball_inside_component(
      labels, DBF, 
      soma_invalidation_scale,
      soma_invalidation_const, 
      anisotropy,
      [root],
      voxel_connectivity_graph=voxel_graph,
    )
  # This target is only valid if no 
  # invalidations have occured yet.
  elif len(manual_targets_before) == 0:
    manual_targets_before.append(target)

  # delete reference to DAF and place it in
  # a list where we can delete it later and
  # free that memory.
  DAF = [ DAF ] 

  paths = compute_paths(
    root, labels, DBF, DAF, 
    parents, scale, const, anisotropy, 
    soma_mode, soma_radius, fix_branching,
    manual_targets_before, manual_targets_after, 
    max_paths, voxel_graph
  )

  skel = Skeleton.simple_merge(
    [ Skeleton.from_path(path) for path in paths if len(path) > 0 ]
  ).consolidate()

  verts = skel.vertices.flatten().astype(np.uint32)
  skel.radii = DBF[verts[::3], verts[1::3], verts[2::3]]

  return skel

def compute_paths(
    root, labels, DBF, DAF, 
    parents, scale, const, anisotropy, 
    soma_mode, soma_radius, fix_branching,
    manual_targets_before, manual_targets_after,
    max_paths, voxel_graph
  ):
  """
  Given the labels, DBF, DAF, dijkstra parents,
  and associated invalidation knobs, find the set of paths 
  that cover the object. Somas are given special treatment
  in that we attempt to cull vertices within a radius of the
  root vertex.
  """
  paths = []
  valid_labels = np.count_nonzero(labels)
  root = tuple(root)

  if max_paths is None:
    max_paths = valid_labels

  if len(manual_targets_before) + len(manual_targets_after) >= max_paths:
    return []

  target_finder = None
  def find_target():
    nonlocal target_finder
    if target_finder is None:
      target_finder = kimimaro.skeletontricks.CachedTargetFinder(labels, DAF[0])
      DAF.pop(0)
    return target_finder.find_target(labels)

  parents[tuple(root)] = 0 # provide initial rail for dijkstra.railroad

  while (valid_labels > 0 or manual_targets_before or manual_targets_after) \
    and len(paths) < max_paths:

    if manual_targets_before:
      target = manual_targets_before.pop()
    elif valid_labels == 0:
      target = manual_targets_after.pop()
    else:
      target = find_target()

    if fix_branching:
      # Draw a path (a "road") from the target to the nearest zero weighted
      # path (a "rail"). This has some minor efficiencies vs drawing
      # from a target all the way to the source. Also, target -> source
      # is much more efficient than source -> target for three reasons.
      # (a) target -> catches a rail instead of exploring all rails
      # (b) target has a natural edge effect that restrict exploration
      # (c) in soma, target -> source follows gradients vs fights them
      path = dijkstra3d.railroad(
        parents, target, voxel_graph=voxel_graph
      )
    else:
      path = dijkstra3d.path_from_parents(parents, target)
    
    if soma_mode:
      dist_to_soma_root = np.linalg.norm(anisotropy * (path - root), axis=1)
      # remove all path points which are within soma_radius of root
      path = np.concatenate(
        (path[:1,:], path[dist_to_soma_root > soma_radius, :])
      )

    if valid_labels > 0:
      invalidated, labels = kimimaro.skeletontricks.roll_invalidation_ball_inside_component(
        labels, DBF, scale, const, 
        anisotropy, path,
        voxel_connectivity_graph=voxel_graph,
      )      
      valid_labels -= invalidated

    for vertex in path:
      if fix_branching:
        parents[tuple(vertex)] = 0.0

    paths.append(path)

  return paths

def find_soma_root(DBF, dbf_max):
  """
  This perhaps overcomplicates things, but it's possible,
  for example in a rectangular cuboid, for there to be
  many multiple maxima at the center of a shape. We pick
  the one closest to the centroid of the shape to ensure
  the choice is sensible.

  Returns: (x,y,z) as integers
  """
  maxima = (DBF == dbf_max)
  com = ndimage.measurements.center_of_mass(maxima)
  com = np.array(com, dtype=np.float32)
  
  coords = np.where(maxima)
  coords = np.vstack( coords ).T
  root = np.argmin(
    np.sum((coords - com) ** 2, axis=1)
  )

  return tuple(coords[root].astype(np.uint32))

def find_root(labels, anisotropy, voxel_graph):
  """
  "4.4 DAF:  Compute distance from any voxel field"
  Compute DAF, but we immediately convert to the PDRF
  The extremal point of the PDRF is a valid root node
  even if the DAF is computed from an arbitrary pixel.
  """
  any_voxel = kimimaro.skeletontricks.first_label(labels)   
  if any_voxel is None: 
    return None

  DAF, target = dijkstra3d.euclidean_distance_field(
    labels, any_voxel, 
    anisotropy=anisotropy,
    return_max_location=True,
    voxel_graph=voxel_graph,
  )
  return target

def is_power_of_two(num):
  if int(num) != num:
    return False
  return num != 0 and ((num & (num - 1)) == 0)

def compute_pdrf(dbf_max, pdrf_scale, pdrf_exponent, DBF, DAF):
  """
  Add p(v) to the DAF (pp. 4, section 4.5)
  "4.5 PDRF: Compute penalized distance from root voxel field"
  Let M > max(DBF)
  p(v) = 5000 * (1 - DBF(v) / M)^16
  5000 is chosen to allow skeleton segments to be up to 3000 voxels
  long without exceeding floating point precision.

  IMPLEMENTATION NOTE: 
  Appearently repeated *= is much faster than "** f(16)" 
  12,740.0 microseconds vs 4 x 560 = 2,240 microseconds (5.69x)

  More clearly written:
  PDRF = DAF + 5000 * ((1 - DBF * M) ** 16)
  """
  f = lambda x: np.float32(x)
  M = f( 1 / (dbf_max ** 1.01) )

  # First branch is much faster than ** which presumably
  # uses logarithms to do the exponentiation.
  if is_power_of_two(pdrf_exponent) and (pdrf_exponent < (2 ** 16)):
    PDRF = (f(1) - (DBF * M)) # ^1
    for _ in range(int(np.log2(pdrf_exponent))):
      PDRF *= PDRF # ^pdrf_exponent
  else: 
    PDRF = (f(1) - (DBF * M)) ** pdrf_exponent

  PDRF *= f(pdrf_scale)

  # provide trickle of gradient so open spaces don't collapse
  max_daf = np.max(DAF)
  if max_daf != 0:
    PDRF += DAF * (1 / max_daf)

  return np.asfortranarray(PDRF)

def point_to_point(
  binary_img, start, end,
  anisotropy=(1,1,1), 
  pdrf_scale=100000, 
  pdrf_exponent=4,
):
  """
  Trace a single centerline path from 
  start to end.
  """
  DBF = edt.edt(
    binary_img, 
    anisotropy=anisotropy,
    black_border=True,
  )
  dbf_max = np.max(DBF)

  DBF = kimimaro.skeletontricks.zero2inf(DBF) # DBF[ DBF == 0 ] = np.inf
  DAF = dijkstra3d.euclidean_distance_field(
    binary_img, start, 
    anisotropy=anisotropy,
  )
  DAF = kimimaro.skeletontricks.inf2zero(DAF) # DAF[ DAF == np.inf ] = 0
  PDRF = compute_pdrf(dbf_max, pdrf_scale, pdrf_exponent, DBF, DAF)
  del DAF

  path = dijkstra3d.dijkstra(PDRF, end, start)
  skel = Skeleton.from_path(path)

  verts = skel.vertices.flatten().astype(np.uint32)
  skel.radii = DBF[verts[::3], verts[1::3], verts[2::3]]
  return skel
