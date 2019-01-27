"""
Certain operations have to be fast for the skeletonization
procedure. The ones that didn't fit elsewhere (e.g. dijkstra
and the euclidean distance transform) have a home here.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018
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
def finite_max(cnp.ndarray[float, cast=True, ndim=3] field):
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

ctypedef fused ALLINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  int8_t
  int16_t
  int32_t
  int64_t

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def get_mapping(
    cnp.ndarray[ALLINT, ndim=3] orig_labels, 
    cnp.ndarray[uint32_t, ndim=3] cc_labels
  ):

  cdef int sx,sy,sz 
  sx = orig_labels.shape[0]
  sy = orig_labels.shape[1]
  sz = orig_labels.shape[2]

  cdef int x,y,z 

  remap = {}

  for z in range(sz):
    for y in range(sy):
      for x in range(sx):
        remap[cc_labels[x,y,z]] = orig_labels[x,y,z]

  return remap


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
  cdef int node
  cdef int child
  cdef int parent
  cdef int depth
  cdef int i

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
