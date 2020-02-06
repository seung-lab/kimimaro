"""
Skeletonization algorithm based on TEASAR (Sato et al. 2000).

Authors: Alex Bae and Will Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institue
Date: June 2018 - April 2019

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
from PIL import Image

import kimimaro.skeletontricks

from cloudvolume import PrecomputedSkeleton, view
from cloudvolume.lib import save_images, mkdir

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
    del DBF
    labels = fill_voids.fill(labels, in_place=True)
    DBF = edt.edt(
      labels, anisotropy=anisotropy, order='F',
      black_border=np.all(labels)
    )
    dbf_max = np.max(DBF) 
    soma_mode = dbf_max > soma_acceptance_threshold

  soma_radius = 0.0

  if root is None:
    if soma_mode:
      root = find_soma_root(DBF, dbf_max)    
      soma_radius = dbf_max * soma_invalidation_scale + soma_invalidation_const
    else:
      root = find_root(labels, anisotropy)
      
  if root is None:
    return PrecomputedSkeleton()
 
  # DBF: Distance to Boundary Field
  # DAF: Distance from any voxel Field (distance from root field)
  # PDRF: Penalized Distance from Root Field
  DBF = kimimaro.skeletontricks.zero2inf(DBF) # DBF[ DBF == 0 ] = np.inf
  DAF = dijkstra3d.euclidean_distance_field(labels, root, anisotropy=anisotropy)
  DAF = kimimaro.skeletontricks.inf2zero(DAF) # DAF[ DAF == np.inf ] = 0
  PDRF = compute_pdrf(dbf_max, pdrf_scale, pdrf_exponent, DBF, DAF)

  # Use dijkstra propogation w/o a target to generate a field of
  # pointers from each voxel to its parent. Then we can rapidly
  # compute multiple paths by simply hopping pointers using path_from_parents
  if not fix_branching:
    parents = dijkstra3d.parental_field(PDRF, root)
    del PDRF
  else:
    parents = PDRF

  if soma_mode:
    invalidated, labels = kimimaro.skeletontricks.roll_invalidation_ball(
      labels, DBF, np.array([root], dtype=np.uint32),
      scale=soma_invalidation_scale,
      const=soma_invalidation_const, 
      anisotropy=anisotropy
    )

  paths = compute_paths(
    root, labels, DBF, DAF, 
    parents, scale, const, pdrf_scale, pdrf_exponent, anisotropy, 
    soma_mode, soma_radius, fix_branching,
    manual_targets_before, manual_targets_after, 
    max_paths
  )

  skel = PrecomputedSkeleton.simple_merge(
    [ PrecomputedSkeleton.from_path(path) for path in paths ]
  ).consolidate()

  verts = skel.vertices.flatten().astype(np.uint32)
  skel.radii = DBF[verts[::3], verts[1::3], verts[2::3]]

  return skel

def compute_paths(
    root, labels, DBF, DAF, 
    parents, scale, const, pdrf_scale, pdrf_exponent, anisotropy, 
    soma_mode, soma_radius, fix_branching,
    manual_targets_before, manual_targets_after,
    max_paths
  ):
  """
  Given the labels, DBF, DAF, dijkstra parents,
  and associated invalidation knobs, find the set of paths 
  that cover the object. Somas are given special treatment
  in that we attempt to cull vertices within a radius of the
  root vertex.
  """
  invalid_vertices = {}
  paths = []
  valid_labels = np.count_nonzero(labels)
  root = tuple(root)

  if soma_mode:
    invalid_vertices[root] = True

  if max_paths is None:
    max_paths = valid_labels

  if len(manual_targets_before) + len(manual_targets_after) >= max_paths:
    return []

  heuristic = 'line' if soma_mode else None

  while (valid_labels > 0 or manual_targets_before or manual_targets_after) \
    and len(paths) < max_paths:

    if manual_targets_before:
      target = manual_targets_before.pop()
    elif valid_labels == 0:
      target = manual_targets_after.pop()
    else:
      target = kimimaro.skeletontricks.find_target(labels, DAF)

    heuristic_norm = -1
    if soma_mode:
      heuristic_norm = compute_heuristic_norm(root, target, pdrf_scale, pdrf_exponent)

    if fix_branching:
      path = dijkstra3d.dijkstra(parents, root, target, heuristic=heuristic, heuristic_args={ 'norm': heuristic_norm })
    else:
      path = dijkstra3d.path_from_parents(parents, target)
    
    if soma_mode:
      dist_to_soma_root = np.linalg.norm(anisotropy * (path - root), axis=1)
      # remove all path points which are within soma_radius of root
      path = np.concatenate(
        (path[:1,:], path[dist_to_soma_root > soma_radius, :])
      )

    if valid_labels > 0:
      invalidated, labels = kimimaro.skeletontricks.roll_invalidation_cube(
        labels, DBF, path, scale, const, 
        anisotropy=anisotropy, invalid_vertices=invalid_vertices,
      )
      valid_labels -= invalidated

    for vertex in path:
      invalid_vertices[tuple(vertex)] = True
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

def find_root(labels, anisotropy):
  """
  "4.4 DAF:  Compute distance from any voxel field"
  Compute DAF, but we immediately convert to the PDRF
  The extremal point of the PDRF is a valid root node
  even if the DAF is computed from an arbitrary pixel.
  """
  any_voxel = kimimaro.skeletontricks.first_label(labels)   
  if any_voxel is None: 
    return None

  DAF = dijkstra3d.euclidean_distance_field(
    np.asfortranarray(labels), any_voxel, anisotropy=anisotropy)
  return kimimaro.skeletontricks.find_target(labels, DAF)

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
  PDRF += DAF * (1 / np.max(DAF)) 

  return np.asfortranarray(PDRF)

def compute_heuristic_norm(root, target, k, p):
  """
  The PDRF is a weirdly shaped function for a simple
  A* search. If we don't compensate for the high
  dynamic range, A* won't really add much. What we do is 
  instead of using the field minimum, we use something 
  approximating the average weight dijkstra will need
  to traverse to get to the target. 

  This is computed as k * sum((x/N)^p) from 1 to N

  Instead of summing over the PDRF in some direction,
  we take N as the distance between the root and target
  and use an analytic form to compute the sum.
  """
  N = np.linalg.norm(np.array(root) - np.array(target))

  evenfn = lambda n: n * (n + 1) * (2 * n + 1)
  oddfn = lambda n: (n * n) * (n + 1) * (n + 1)

  powersumfn = {
    0: (lambda n: n),
    1: (lambda n: n * (n + 1) / 2),
    2: (lambda n: evenfn(n) / 6),
    3: (lambda n: oddfn(n) / 4),
    4: (lambda n: evenfn(n) * (3 * n * n + 3 * n - 1) / 30),
    5: (lambda n: oddfn(n) * (2 * n * n * 2 * n - 1) / 12),
    6: (lambda n: evenfn(n) * (3 * n ** 4 + 6 * n ** 3 - 3 * n + 1) / 42),
    7: (lambda n: oddfn(n) * ((3 * n ** 4) + (6 * n ** 3) - (n ** 2) - (4 * n) + 2) / 24),
    8: (
      lambda n: evenfn(n) * ((5 * n ** 6) + (15 * n ** 5) \
        + (5 * n ** 4) - (15 * n ** 3) - n * n + 9 * n - 3) / 90
    ),
    9: (
      lambda n: oddfn(n) * (n * n + n - 1) \
        * (2 * n ** 4 + 4 * n ** 3 - n * n - 3 * n + 3) / 20
    ),
    # 10: (lambda n: ...)
    # 11: (lambda n: ...)
    # 12: (lambda n: ...)
    # 13: (lambda n: ...)
    # 14: (lambda n: ...)
    # 15: (lambda n: ...)
    16: (
      lambda n: evenfn(n) * ((15 * n ** 14) + (105 * n ** 13) \
        + (175 * n ** 12) - (315 * n ** 11) - (805 * n ** 10) + (1365 * n ** 9) \
        + (2775 * n ** 8) - (4845 * n ** 7) - (6275 * n ** 6) + (11835 * n ** 5) \
        + (7485 * n ** 4) - (17145 * n ** 3) - (1519 * n * n) + (10851 * n) - 3617) / 510
    ),
  }

  return powersumfn[p](N) / (N ** p) * k