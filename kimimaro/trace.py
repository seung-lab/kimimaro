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
    manual_targets=[],
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
  manual_targets: list of (x,y,z) that coorrespond to locations that must 
    have paths drawn to. Used for specifying root and border targets for
    merging adjacent chunks out-of-core.
  max_paths: If a label requires drawing this number of paths or more,
    abort and move onto the next label.

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
    labels = ndimage.binary_fill_holes(labels)
    labels = np.asfortranarray(labels)
    DBF = edt.edt(labels, anisotropy=anisotropy, order='F')   
    dbf_max = np.max(DBF)
    soma_mode = dbf_max > soma_acceptance_threshold

  if soma_mode:
    root = np.unravel_index(np.argmax(DBF), DBF.shape)
    soma_radius = dbf_max * soma_invalidation_scale + soma_invalidation_const
  else:
    root = find_root(labels, manual_targets, anisotropy)
    soma_radius = 0.0

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
    parents, scale, const, anisotropy, 
    soma_mode, soma_radius, fix_branching,
    manual_targets, max_paths
  )

  skel = PrecomputedSkeleton.simple_merge(
    [ PrecomputedSkeleton.from_path(path) for path in paths ]
  ).consolidate()

  verts = skel.vertices.flatten().astype(np.uint32)
  skel.radii = DBF[verts[::3], verts[1::3], verts[2::3]]

  return skel

def compute_paths(
    root, labels, DBF, DAF, 
    parents, scale, const, anisotropy, 
    soma_mode, soma_radius, fix_branching,
    manual_targets, max_paths
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

  if soma_mode:
    invalid_vertices[root] = True

  if max_paths is None:
    max_paths = valid_labels

  if len(manual_targets) >= max_paths:
    return []

  while (valid_labels > 0 or manual_targets) and len(paths) < max_paths:
    if manual_targets:
      target = manual_targets.pop()
    else:
      target = kimimaro.skeletontricks.find_target(labels, DAF)

    if fix_branching:
      path = dijkstra3d.dijkstra(parents, root, target)
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

def find_root(labels, manual_targets, anisotropy):
  """
  "4.4 DAF:  Compute distance from any voxel field"
  Compute DAF, but we immediately convert to the PDRF
  The extremal point of the PDRF is a valid root node
  even if the DAF is computed from an arbitrary pixel.
  """
  if len(manual_targets):
    return manual_targets.pop()

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

def xy_path_projection(paths, labels, N=0):
  """Used for debugging paths."""
  if type(paths) != list:
    paths = [ paths ]

  projection = np.zeros( (labels.shape[0], labels.shape[1] ), dtype=np.uint8)
  outline = labels.any(axis=-1).astype(np.uint8) * 77
  outline = outline.reshape( (labels.shape[0], labels.shape[1] ) )
  projection += outline
  for path in paths:
    for coord in path:
      projection[coord[0], coord[1]] = 255

  projection = Image.fromarray(projection.T, 'L')
  N = str(N).zfill(3)
  mkdir('./saved_images/projections')
  projection.save('./saved_images/projections/{}.png'.format(N), 'PNG')

