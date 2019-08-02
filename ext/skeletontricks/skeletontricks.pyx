"""
Certain operations have to be fast for the skeletonization
procedure. The ones that didn't fit elsewhere (e.g. dijkstra
and the euclidean distance transform) have a home here.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018

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
cimport numpy as cnp
import numpy as np

from collections import defaultdict

cdef extern from "math.h":
  float INFINITY

ctypedef fused INTEGER: 
  int8_t
  int16_t
  int32_t
  int64_t
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  unsigned char

cdef extern from "skeletontricks.hpp" namespace "skeletontricks":
  cdef int _roll_invalidation_cube(
    uint8_t* labels, float* DBF,
    int sx, int sy, int sz,
    float wx, float wy, float wz,
    int* path, int path_size,
    float scale, float constant
  )

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def inf2zero(cnp.ndarray[float, cast=True, ndim=3] field):
  """
  inf2zero(cnp.ndarray[float, cast=True, ndim=3] field)

  Convert infinities to zeros.

  Returns: field
  """
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

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
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

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
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

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
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

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
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

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
  cdef int x,y,z
  cdef int sx, sy, sz

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef int mx, my, mz

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
  cdef int sx, sy, sz 
  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef float wx, wy, wz
  (wx, wy, wz) = anisotropy
    
  cdef float radius, dist
  cdef int minx, maxx, miny, maxy, minz, maxz

  cdef int x,y,z
  cdef int x0, y0, z0

  cdef int invalidated = 0

  for coord in path:
    if tuple(coord) in invalid_vertices:
      continue

    (x0, y0, z0) = coord
    radius = DBF[x0,y0,z0] * scale + const # physical units (e.g. nm)

    minx = max(0,  <int>(0.5 + (x0 - (radius / wx))))
    maxx = min(sx, <int>(0.5 + (x0 + (radius / wx))))
    miny = max(0,  <int>(0.5 + (y0 - (radius / wy))))
    maxy = min(sy, <int>(0.5 + (y0 + (radius / wy))))
    minz = max(0,  <int>(0.5 + (z0 - (radius / wz))))
    maxz = min(sz, <int>(0.5 + (z0 + (radius / wz))))

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
    cnp.ndarray[uint32_t, ndim=3] cc_labels
  ):
  """
  get_mapping(
    ndarray[INTEGER, ndim=3] orig_labels, 
    ndarray[uint32_t, ndim=3] cc_labels
  )

  Given a set of possibly not connected labels 
  and an image containing their labeled connected components, 
  produce a dictionary containing the inverse of this mapping.

  Returns: { $CC_LABEL: $ORIGINAL_LABEL }
  """

  cdef int sx,sy,sz 
  sx = orig_labels.shape[0]
  sy = orig_labels.shape[1]
  sz = orig_labels.shape[2]

  cdef int x,y,z 

  remap = {}

  if orig_labels.size == 0:
    return remap

  cdef uint32_t last_label = cc_labels[0,0,0]
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
    cnp.ndarray[uint32_t, ndim=2] labels,
    float wx, float wy
  ):
  """
  compute_centroids(
    cnp.ndarray[uint32_t, ndim=2] labels,
    float wx, float wy
  )

  Compute the centroid for every label on a 2D image at once.

  Returns: { $segid: (x, y), ... }
  """

  cdef float[:] xsum = np.zeros( (labels.size,), dtype=np.float32)
  cdef float[:] ysum = np.zeros( (labels.size,), dtype=np.float32)
  cdef uint32_t[:] labelct = np.zeros( (labels.size,), dtype=np.uint32)

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
    cnp.ndarray[uint32_t, ndim=2] cc_labels,
    float wx, float wy
  ):
  """
  find_border_targets(
    ndarray[float, ndim=2] dt, 
    ndarray[uint32_t, ndim=2] cc_labels,
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

  cdef uint32_t label = 0
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
  cdef int sx, sy, sz 
  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef int sxy = sx * sy

  cdef float wx, wy, wz
  (wx, wy, wz) = anisotropy

  path = [ 
    coord[0] + sx * coord[1] + sxy * coord[2] 
    for coord in path if tuple(coord) not in invalid_vertices 
  ]
  path = np.array(path, dtype=np.int32)

  cdef int[:] pathview = path

  cdef int invalidated = _roll_invalidation_cube(
    <uint8_t*>&labels[0,0,0], <float*>&DBF[0,0,0],
    sx, sy, sz, 
    wx, wy, wz,
    <int*>&pathview[0], path.size,
    scale, const
  )

  return invalidated, labels

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def unique(cnp.ndarray[INTEGER, ndim=3] labels, return_counts=False):
  """
  unique(cnp.ndarray[INTEGER, ndim=3] labels, return_counts=False)

  Faster implementation of np.unique that depends
  on the maximum label in the array being less than
  the size of the array.
  """
  cdef size_t max_label = np.max(labels)

  cdef cnp.ndarray[uint32_t, ndim=1] counts = np.zeros( 
    (max_label+1,), dtype=np.uint32
  )

  cdef size_t x, y, z
  cdef size_t sx = labels.shape[0]
  cdef size_t sy = labels.shape[1]
  cdef size_t sz = labels.shape[2]

  if labels.flags['C_CONTIGUOUS']:
    for x in range(sx):
      for y in range(sy):
        for z in range(sz):
          counts[labels[x,y,z]] += 1
  else:
    for z in range(sz):
      for y in range(sy):
        for x in range(sx):
          counts[labels[x,y,z]] += 1

  cdef list segids = []
  cdef list cts = []

  cdef size_t i = 0
  for i in range(max_label + 1):
    if counts[i] > 0:
      segids.append(i)
      cts.append(counts[i])

  if return_counts:
    return segids, cts
  else:
    return segids

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def find_cycle(cnp.ndarray[int32_t, ndim=2] edges):
  """
  Given a graph of edges that are a single connected component,
  find a cycle via depth first search.

  Returns: list of edges in a cycle (empty list if no cycle is found)
  """
  index = defaultdict(set)
  visited = defaultdict(int)

  if edges.size == 0:
    return []

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
    return []
  
  for i in range(len(path) - 1):
    if path[i] == node:
      break

  path = path[i:]

  if len(path) < 3:
    return []

  cdef list elist = []
  for i in range(len(path) - 1):
    elist.append(
      (path[i], path[i+1])
    )

  return elist
