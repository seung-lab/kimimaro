"""
Certain operations have to be fast for the skeletonization
procedure. The ones that didn't fit elsewhere have a home here.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018 - Februrary 2020

*****************************************************************
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
*****************************************************************
"""
cimport cython
from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t
)
from libcpp cimport bool
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair as cpp_pair

cimport numpy as cnp
import numpy as np

from collections import defaultdict

cdef extern from "math.h":
  float INFINITY

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  unsigned char

ctypedef fused INTEGER: 
  int8_t
  int16_t
  int32_t
  int64_t
  UINT

cdef extern from "skeletontricks.hpp" namespace "skeletontricks":
  cdef size_t _roll_invalidation_cube(
    uint8_t* labels, float* DBF,
    int64_t sx, int64_t sy, int64_t sz,
    float wx, float wy, float wz,
    size_t* path, size_t path_size,
    float scale, float constant
  )

  cdef vector[T] _find_cycle[T](T* edges, size_t Ne)
  
  cdef unordered_map[ uint64_t, float ] _create_distance_graph(
    float* vertices, size_t Nv, 
    uint32_t* edges, size_t Ne, uint32_t start_node,
    vector[int32_t] critical_points_vec
  )

def find_cycle(cnp.ndarray[int32_t, ndim=2] edges):
  """
  Given a graph of edges that are a single connected component,
  find a cycle via depth first search.

  Returns: list of edges in a cycle (empty list if no cycle is found)
  """
  if edges.size == 0:
    return np.zeros((0,), dtype=np.uint32)

  edges = np.ascontiguousarray(edges)

  cdef cnp.ndarray[int32_t, ndim=1] elist = np.array(
    _find_cycle[int32_t](
      <int32_t*>&edges[0,0], <size_t>(edges.size // 2)
    ),
    dtype=np.int32
  )
  return elist

def create_distance_graph(skeleton):
  """
  Creates the distance "supergraph" from a single connected component 
  skeleton as described in _remove_ticks.

  Returns: a distance "supergraph" describing the physical distance
    between the critical points in the skeleton's structure.

  Example skeleton with output:

      60nm   60nm   60nm     
    1------2------3------4
      30nm |  70nm \
           5        ----6

  { 
    (1,2): 60,  
    (2,3): 60,
    (2,5): 30,
    (3,4): 60,
    (3,6): 70,
  }
  """
  cdef cnp.ndarray[float, ndim=2] vertices = skeleton.vertices
  cdef cnp.ndarray[uint32_t, ndim=2] edges = skeleton.edges

  unique_nodes, unique_counts = np.unique(edges, return_counts=True)
  terminal_nodes = unique_nodes[ unique_counts == 1 ]
  branch_nodes = set(unique_nodes[ unique_counts >= 3 ])
  
  critical_points = set(terminal_nodes)
  critical_points.update(branch_nodes)

  res = _create_distance_graph(
    <float*>&vertices[0,0], vertices.shape[0],
    <uint32_t*>&edges[0,0], edges.shape[0], terminal_nodes[0],
    list(critical_points)
  )
  cdef dict supergraph = res

  cdef dict real_supergraph = {}
  cdef uint64_t key = 0
  cdef int32_t e1, e2

  for key in supergraph.keys():
    e2 = <int32_t>(key & 0xffffffff)
    e1 = <int32_t>(key >> 32)
    real_supergraph[ (e1, e2) ] = supergraph[key]

  return real_supergraph


@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def inf2zero(cnp.ndarray[float, cast=True, ndim=3] field):
  """
  inf2zero(cnp.ndarray[float, cast=True, ndim=3] field)

  Convert infinities to zeros.

  Returns: field
  """
  cdef size_t sx, sy, sz 
  cdef size_t  x,  y,  z

  sx = field.shape[0]
  sy = field.shape[1]
  sz = field.shape[2]

  for z in range(0, sz):
    for y in range(0, sy):
      for x in range(0, sx):
        if (field[x,y,z] == INFINITY):
          field[x,y,z] = 0

  return field

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def zero2inf(cnp.ndarray[float, cast=True, ndim=3] field):
  """
  zero2inf(cnp.ndarray[float, cast=True, ndim=3] field)

  Convert zeros to positive infinities.

  Returns: field
  """
  cdef size_t sx, sy, sz 
  cdef size_t  x,  y,  z

  sx = field.shape[0]
  sy = field.shape[1]
  sz = field.shape[2]

  for z in range(0, sz):
    for y in range(0, sy):
      for x in range(0, sx):
        if (field[x,y,z] == 0):
          field[x,y,z] = INFINITY

  return field

@cython.boundscheck(False)  
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
@cython.nonecheck(False)  
def zero_out_all_except(cnp.ndarray[INTEGER, cast=True, ndim=3] field, INTEGER leave_alone): 
  """
  zero_out_all_except(cnp.ndarray[INTEGER, cast=True, ndim=3] field, INTEGER leave_alone)

  Change all values in field to zero except `leave_alone`.

  Returns: field
  """
  cdef size_t sx, sy, sz   
  cdef size_t  x,  y,  z 

  sx = field.shape[0]  
  sy = field.shape[1] 
  sz = field.shape[2] 

  for z in range(0, sz): 
    for y in range(0, sy):  
      for x in range(0, sx):  
        if (field[x,y,z] != leave_alone): 
          field[x,y,z] = 0  

  return field  

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def finite_max(cnp.ndarray[float, cast=True, ndim=3] field):
  """
  float finite_max(cnp.ndarray[float, cast=True, ndim=3] field)

  Given a field of floats that may include infinities, find the 
  largest finite value.
  """
  cdef size_t sx, sy, sz 
  cdef size_t  x,  y,  z

  sx = field.shape[0]
  sy = field.shape[1]
  sz = field.shape[2]

  cdef float maximum = -INFINITY
  for z in range(0, sz):
    for y in range(0, sy):
      for x in range(0, sx):
        if (field[x,y,z] > maximum) and (field[x,y,z] < +INFINITY):
          maximum = field[x,y,z]

  return maximum

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def finite_min(cnp.ndarray[float, cast=True, ndim=3] field):
  """
  float finite_min(cnp.ndarray[float, cast=True, ndim=3] field)

  Given a field of floats that may include infinities, find the 
  minimum finite value.
  """
  cdef size_t sx, sy, sz 
  cdef size_t  x,  y,  z

  sx = field.shape[0]
  sy = field.shape[1]
  sz = field.shape[2]

  cdef float minimum = -INFINITY
  for z in range(0, sz):
    for y in range(0, sy):
      for x in range(0, sx):
        if (field[x,y,z] < minimum) and (field[x,y,z] > -INFINITY):
          minimum = field[x,y,z]

  return minimum

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def first_label(cnp.ndarray[uint8_t, cast=True, ndim=3] labels):
  """
  uint8_t first_label(cnp.ndarray[uint8_t, cast=True, ndim=3] labels)

  Scan through labels to find the first non-zero value and return it.
  """
  cdef size_t sx, sy, sz 
  cdef size_t  x,  y,  z

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  for z in range(0, sz):
    for y in range(0, sy):
      for x in range(0, sx):
        if labels[x,y,z]:
          return (x,y,z)

  return None

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def find_target(
    cnp.ndarray[uint8_t, cast=True, ndim=3] labels, 
    cnp.ndarray[float, ndim=3] PDRF
  ):
  """
  find_target(ndarray[uint8_t, cast=True, ndim=3] labels, ndarray[float, ndim=3] PDRF)

  Given a binary image and a coregistered map of values to it, 
  find the coordinate of the voxel corresponding to the first
  instance of the maximum map value.

  Returns: (x, y, z)
  """
  cdef size_t x,y,z
  cdef size_t sx, sy, sz

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef int64_t mx, my, mz

  mx = -1
  my = -1
  mz = -1

  cdef float maxpdrf = -INFINITY
  for x in range(0, sx):
    for y in range(0, sy):
      for z in range(0, sz):
        if labels[x,y,z] and PDRF[x,y,z] > maxpdrf:
          maxpdrf = PDRF[x,y,z]
          mx = x
          my = y
          mz = z

  return (mx, my, mz)


@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def roll_invalidation_ball(
    cnp.ndarray[uint8_t, cast=True, ndim=3] labels, 
    cnp.ndarray[float, ndim=3] DBF, 
    path, float scale, float const,
    anisotropy=(1,1,1),
    invalid_vertices={},
  ):
  """
  roll_invalidation_ball(
    ndarray[uint8_t, cast=True, ndim=3] labels, ndarray[float, ndim=3] DBF, 
    path, float scale, float const, anisotropy=(1,1,1), invalid_vertices={}
  )

  Given an anisotropic binary image, its distance transform, and a path 
  traversing the binary image, erase the voxels surrounding the path
  in a sphere around each vertex on the path corresponding to the 
  equation: 

  r = scale * DBF[x,y,z] + const

  Returns: modified labels
  """
  cdef int64_t sx, sy, sz 
  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef float wx, wy, wz
  (wx, wy, wz) = anisotropy
    
  cdef float radius, dist
  cdef int64_t minx, maxx, miny, maxy, minz, maxz

  cdef int64_t x,y,z
  cdef int64_t x0, y0, z0

  cdef size_t invalidated = 0

  for coord in path:
    if tuple(coord) in invalid_vertices:
      continue

    (x0, y0, z0) = coord
    radius = DBF[x0,y0,z0] * scale + const # physical units (e.g. nm)

    minx = max(0,  <int64_t>(0.5 + (x0 - (radius / wx))))
    maxx = min(sx, <int64_t>(0.5 + (x0 + (radius / wx))))
    miny = max(0,  <int64_t>(0.5 + (y0 - (radius / wy))))
    maxy = min(sy, <int64_t>(0.5 + (y0 + (radius / wy))))
    minz = max(0,  <int64_t>(0.5 + (z0 - (radius / wz))))
    maxz = min(sz, <int64_t>(0.5 + (z0 + (radius / wz))))

    radius *= radius 

    for x in range(minx, maxx):
      for y in range(miny, maxy):
        for z in range(minz, maxz):
          if not labels[x,y,z]:
            continue 

          dist = (wx * (x - x0)) ** 2 + (wy * (y - y0)) ** 2 + (wz * (z - z0)) ** 2
          if dist <= radius:
            invalidated += 1
            labels[x,y,z] = 0

  return invalidated, labels

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def get_mapping(
    cnp.ndarray[INTEGER, ndim=3] orig_labels, 
    cnp.ndarray[UINT, ndim=3] cc_labels
  ):
  """
  get_mapping(
    ndarray[INTEGER, ndim=3] orig_labels, 
    ndarray[UINT, ndim=3] cc_labels
  )

  Given a set of possibly not connected labels 
  and an image containing their labeled connected components, 
  produce a dictionary containing the inverse of this mapping.

  Returns: { $CC_LABEL: $ORIGINAL_LABEL }
  """

  cdef size_t sx, sy, sz 
  sx = orig_labels.shape[0]
  sy = orig_labels.shape[1]
  sz = orig_labels.shape[2]

  cdef size_t x,y,z 

  remap = {}

  if orig_labels.size == 0:
    return remap

  cdef UINT last_label = cc_labels[0,0,0]
  remap[cc_labels[0,0,0]] = orig_labels[0,0,0]

  for z in range(sz):
    for y in range(sy):
      for x in range(sx):
        if last_label == cc_labels[x,y,z]:
          continue
        remap[cc_labels[x,y,z]] = orig_labels[x,y,z]
        last_label = cc_labels[x,y,z]

  return remap

def compute_centroids(
    cnp.ndarray[UINT, ndim=2] labels,
    float wx, float wy
  ):
  """
  compute_centroids(
    cnp.ndarray[UINT, ndim=2] labels,
    float wx, float wy
  )

  Compute the centroid for every label on a 2D image at once.

  Returns: { $segid: (x, y), ... }
  """

  cdef float[:] xsum = np.zeros( (labels.size,), dtype=np.float32)
  cdef float[:] ysum = np.zeros( (labels.size,), dtype=np.float32)
  cdef UINT[:] labelct = np.zeros( (labels.size,), dtype=labels.dtype)

  cdef size_t sx, sy
  sx = labels.shape[0]
  sy = labels.shape[1]

  cdef size_t x, y
  cdef uint32_t label = 0

  for x in range(sx):
    for y in range(sy):
      label = labels[x,y]
      if label == 0:
        continue

      xsum[label] += x 
      ysum[label] += y 
      labelct[label] += 1

  result = {}

  cdef float cx = wx * sx / 2
  cdef float cy = wy * sy / 2

  cdef float px, py

  for label in range(labels.size):
    if labelct[label] == 0:
      continue

    px = wx * <float>xsum[label] / <float>labelct[label]
    py = wy * <float>ysum[label] / <float>labelct[label]

    # Since we don't know which coordinate frame we 
    # are using, round toward the center of the image
    # to ensure we get the same pixel every time.
    if px - cx >= 0:
      px = px # will be truncated towards center
    else:
      px = px + wx

    if py - cy >= 0:
      py = py # will be truncated towards center
    else:
      py = py + wy

    result[label] = (<int>(px / wx), <int>(py / wy))

  return result

def find_border_targets(
    cnp.ndarray[float, ndim=2] dt,
    cnp.ndarray[UINT, ndim=2] cc_labels,
    float wx, float wy
  ):
  """
  find_border_targets(
    ndarray[float, ndim=2] dt, 
    ndarray[UINT, ndim=2] cc_labels,
    float wx, float wy
  )

  Given a set of connected components that line within 
  a plane and their distance transform, return a map of
  label ID to the coordinate of its maximum distance 
  transform value. If there are multiple maxima, we 
  disambiguate based on topological criteria that are
  coordinate frame independent in order to avoid dealing
  with issues that come from the six rotated frames and
  their mirrored partners.

  The purpose of this function is to fix the edge effect
  the standard TEASAR algorithm generates and ensure that
  we can trivially join skeletons from adjacent chunks.  

  Rotating the (x,y) pairs into their appropriate frame
  is performed in the function that calls this one.

  Returns: { $SEGID: (x, y), ... }
  """
  cdef size_t sx, sy
  sx = dt.shape[0]
  sy = dt.shape[1]

  cdef size_t x, y

  mx = defaultdict(float)
  pts = {}

  cdef UINT label = 0
  cdef dict centroids = compute_centroids(cc_labels, wx, wy)

  cdef float px, py
  cdef float centx, centy

  for y in range(sy):
    for x in range(sx):
      label = cc_labels[x,y]
      if label == 0:
        continue
      elif dt[x,y] == 0:
        continue
      elif dt[x,y] > mx[label]:
        mx[label] = dt[x,y]
        pts[label] = (x,y)
      elif mx[label] == dt[x,y]:
        px, py = pts[label]
        centx, centy = centroids[label]
        pts[label] = compute_tiebreaker_maxima(
          px, py, x, y, 
          centx, centy,
          sx, sy, wx, wy
        )

  return pts

def compute_tiebreaker_maxima(
    float px, float py, 
    float x, float y, 
    float centx, float centy,
    float sx, float sy,
    float wx, float wy
  ):
  """
  compute_tiebreaker_maxima(
    float px, float py, 
    float x, float y, 
    float centx, float centy,
    float sx, float sy,
    float wx, float wy
  )

  This function breaks ties for `compute_border_targets`.

  (px,py): A previously found distance transform maxima 
  (x,y): The coordinate of the newly found maxima
  (sx,sy): The length and width of the image plane.
  (wx,wy): Weighting for anisotropy.
  (centx, centy): The centroid of the current label.

  We use following topolological criteria to achieve
  a coordinate frame-free voxel selection. We pick
  the result of the first criterion that is satisfied.

  1) Pick the voxel closest to the centroid of the label.
  2) The voxel closest to the centroid of the plane.
  3) Closest to a corner of the plane.
  4) Closest to an edge of the plane.
  5) The previous maxima.

  The worst case would be an annulus drawn around the center,
  which would result in four equally eligible pixels....

  Hopefully this won't happen too often...

  Returns: some (x, y)
  """
  cdef float cx = wx * sx / 2.0
  cdef float cy = wy * sy / 2.0

  cdef float dist1 = distsq(px,py, centx,centy, wx,wy)
  cdef float dist2 = distsq( x, y, centx,centy, wx,wy)

  if dist2 < dist1:
    return (x, y)
  elif dist1 == dist2:
    dist1 = distsq(px,py, cx,cy, wx,wy)
    dist2 = distsq( x, y, cx,cy, wx,wy)
    if dist2 < dist1:
      return (x,y)
    elif dist1 == dist2:
      dist1 = cornerness(px, py, sx, sy, wx,wy)
      dist2 = cornerness( x,  y, sx, sy, wx,wy)
      if dist2 < dist1:
        return (x, y)
      elif dist1 == dist2:
        dist1 = edgeness(px, py, sx, sy, wx,wy)
        dist2 = edgeness( x,  y, sx, sy, wx,wy)
        if dist2 < dist1:
          return (x, y)

  return (px, py)

cdef float edgeness(
    float x, float y, float sx, float sy,
    float wx, float wy
  ):
  """
  float edgeness(float x, float y, float sx, float sy)

  Nearness of (x,y) to the edge of an image of size (sx,sy).
  """
  return min(
    wx * (x - 0.5),
    wx * (sx - 0.5 - x),
    wy * (y - 0.5),
    wy * (sy - 0.5 - y)
  )

cdef float cornerness(
    float x, float y, float sx, float sy,
    float wx, float wy
  ):
  """
  float cornerness(
      float x, float y, float sx, float sy
      float wx, float wy
  )

  Nearness of (x,y) to a corner of an image of size (sx,sy).
  """
  return min( 
    distsq(x,y,-0.5,-0.5, wx, wy), 
    distsq(x,y,sx-0.5,-0.5, wx, wy),
    distsq(x,y,sx-0.5,sy-0.5, wx, wy),
    distsq(x,y,-0.5,sx-0.5, wx, wy)
  )

cdef float distsq(
    float p1x, float p1y, 
    float p2x, float p2y, 
    float wx, float wy
  ):

  p1x = wx * (p1x - p2x)
  p1y = wy * (p1y - p2y)
  return p1x * p1x + p1y * p1y 

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def roll_invalidation_cube(
    cnp.ndarray[uint8_t, cast=True, ndim=3] labels, 
    cnp.ndarray[float, ndim=3] DBF, 
    path, float scale, float const,
    anisotropy=(1,1,1),
    invalid_vertices={},
  ):
  """
  roll_invalidation_cube(
    ndarray[uint8_t, cast=True, ndim=3] labels, 
    ndarray[float, ndim=3] DBF, 
    path, float scale, float const,
    anisotropy=(1,1,1),
    invalid_vertices={},
  )

  Given an anisotropic binary image, its distance transform, and a path 
  traversing the binary image, erase the voxels surrounding the path
  in a cube around each vertex. In contrast to `roll_invalidation_ball`,
  this function runs in time linear in the number of image pixels.
  """
  cdef int64_t sx, sy, sz 
  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef size_t sxy = sx * sy

  cdef float wx, wy, wz
  (wx, wy, wz) = anisotropy

  path = [ 
    coord[0] + sx * coord[1] + sxy * coord[2] 
    for coord in path if tuple(coord) not in invalid_vertices 
  ]
  path = np.array(path, dtype=np.uintp)

  cdef size_t[:] pathview = path

  cdef size_t invalidated = _roll_invalidation_cube(
    <uint8_t*>&labels[0,0,0], <float*>&DBF[0,0,0],
    sx, sy, sz, 
    wx, wy, wz,
    <size_t*>&pathview[0], path.size,
    scale, const
  )

  return invalidated, labels

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def find_cycle_cython(cnp.ndarray[int32_t, ndim=2] edges):
  """
  Given a graph of edges that are a single connected component,
  find a cycle via depth first search.

  Returns: list of edges in a cycle (empty list if no cycle is found)
  """
  index = defaultdict(set)
  visited = defaultdict(int)

  if edges.size == 0:
    return np.array([], dtype=np.int32)

  for e1, e2 in edges:
    index[e1].add(e2)
    index[e2].add(e1)

  cdef int root = edges[0,0]
  cdef int node = -1
  cdef int child = -1
  cdef int parent = -1
  cdef int depth = -1
  cdef int i = 0

  cdef list stack = [root]
  cdef list parents = [-1]
  cdef list depth_stack = [0]
  cdef list path = []

  while stack:
    node = stack.pop()
    parent = parents.pop()
    depth = depth_stack.pop()

    for i in range(len(path) - depth):
      path.pop()

    path.append(node)

    if visited[node] == 1:
      break

    visited[node] = 1

    for child in index[node]:
      if child != parent:
        stack.append(child)
        parents.append(node)
        depth_stack.append(depth + 1)

  if len(path) <= 1:
    return np.array([], dtype=np.int32)
  
  for i in range(len(path) - 1):
    if path[i] == node:
      break

  path = path[i:]

  if len(path) < 3:
    return np.array([], dtype=np.int32)

  return np.array(path, dtype=np.int32)

def find_avocado_fruit(
  cnp.ndarray[INTEGER, ndim=3] labels, 
  size_t cx, size_t cy, size_t cz,
  INTEGER background = 0
):
  """
  Tests to see if the current coordinate is inside 
  the nucleus of a somata that has been assigned
  to a separate label from the rest of the cell.

  Returns: (pit, fruit)
  """
  cdef size_t sx, sy, sz
  sx, sy, sz = labels.shape[:3]
  cdef size_t voxels = sx * sy * sz 

  if cx >= sx or cy >= sy or cz >= sz:
    raise ValueError(
      "<{},{},{}> must be be contained within shape <{},{},{}>".format(
        cx,cy,cz,sx,sy,sz
    ))

  cdef size_t x, y, z 
  cdef INTEGER label = labels[cx, cy, cz]
  cdef list changes = [ None ] * 6

  for x in range(cx, sx):
    if labels[x,cy,cz] == background:
      break
    elif labels[x,cy,cz] != label:
      changes[0] = labels[x,cy,cz]
      break

  for x in range(cx, 0, -1):
    if labels[x,cy,cz] == background:
      break
    elif labels[x,cy,cz] != label:
      changes[1] = labels[x,cy,cz]
      break

  for y in range(cy, sy):
    if labels[cx,y,cz] == background:
      break
    if labels[cx,y,cz] != label:
      changes[2] = labels[cx,y,cz]
      break

  for y in range(cy, 0, -1):
    if labels[cx,y,cz] == background:
      break
    if labels[cx,y,cz] != label:
      changes[3] = labels[cx,y,cz]
      break

  for z in range(cz, sz):
    if labels[cx,cy,z] == background:
      break
    if labels[cx,cy,z] != label:
      changes[4] = labels[cx,cy,z]
      break

  for z in range(cz, 0, -1):
    if labels[cx,cy,z] == background:
      break
    if labels[cx,cy,z] != label:
      changes[5] = labels[cx,cy,z]
      break

  changes = [ _ for _ in changes if _ is not None ]

  if len(changes) == 0:
    return (label, label)

  if len(changes) > 3: # if more than 3, allow one non-match
    allowed_differences = 1
  else: # allow no non-matches (we're in a corner)
    allowed_differences = 0

  uniq, cts = np.unique(changes, return_counts=True)
  candidate_fruit_index = np.argmax(cts)
  differences = len(changes) - cts[candidate_fruit_index]

  # it's not an avocado if there's lots of
  # labels surrounding the candidate "pit"
  if differences > allowed_differences:
    return (label, label)
  
  return (label, uniq[candidate_fruit_index])

  


