import pytest

import edt
import numpy as np
from osteoid import Skeleton

import kimimaro.intake
import kimimaro.post
import kimimaro.skeletontricks
from kimimaro.utility import moving_average, cross_sectional_area

@pytest.fixture
def connectomics_data():
  import crackle
  return crackle.load("benchmarks/connectomics.npy.ckl.gz")

def test_empty_image():
  labels = np.zeros( (256, 256, 256), dtype=bool)  
  skels = kimimaro.skeletonize(labels, fix_borders=True)

  assert len(skels) == 0

def test_very_sparse_image():
  labels = np.zeros( (64, 64, 64), dtype=bool)  
  labels[5,5,5] = True
  labels[6,5,5] = True
  labels[20,20,20] = True 
  skels = kimimaro.skeletonize(labels, dust_threshold=0)
  
  # single voxels don't get skeletonized
  assert len(skels) == 1

def test_solid_image():
  labels = np.ones( (128, 128, 128), dtype=bool)  
  skels = kimimaro.skeletonize(labels, fix_borders=True)

  assert len(skels) == 1

def test_binary_image():
  labels = np.ones( (256, 256, 3), dtype=bool)
  labels[-1,0] = 0
  labels[0,-1] = 0
  
  skels = kimimaro.skeletonize(labels, fix_borders=False)

  assert len(skels) == 1

@pytest.mark.parametrize('fill_holes', (True, False))
def test_square(fill_holes):
  labels = np.ones( (1000, 1000), dtype=np.uint8)
  labels[-1,0] = 0
  labels[0,-1] = 0
  
  teasar_params = {
    "scale": 1.5, 
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_const": 300,
    "soma_invalidation_scale": 2
  }

  skels = kimimaro.skeletonize(labels, teasar_params=teasar_params, fix_borders=False, fill_holes=fill_holes)

  assert len(skels) == 1

  skel = skels[1]
  assert skel.vertices.shape[0] == 1000
  assert skel.edges.shape[0] == 999
  assert abs(skel.cable_length() - 999 * np.sqrt(2)) < 0.001

  labels = np.ones( (1000, 1000), dtype=np.uint8)
  labels[0,0] = 0
  labels[-1,-1] = 0

  skels = kimimaro.skeletonize(labels, teasar_params=teasar_params, fix_borders=False, fill_holes=fill_holes)

  assert len(skels) == 1

  skel = skels[1]
  assert skel.vertices.shape[0] == 1000
  assert skel.edges.shape[0] == 999
  assert abs(skel.cable_length() - 999 * np.sqrt(2)) < 0.001

def test_cube():
  labels = np.ones( (128, 128, 128), dtype=np.uint8)
  labels[0, 0, 0] = 0
  labels[-1, -1, -1] = 0
  
  skels = kimimaro.skeletonize(labels, fix_borders=False)

  assert len(skels) == 1

  skel = skels[1]
  assert skel.vertices.shape[0] == 128
  assert skel.edges.shape[0] == 127
  assert abs(skel.cable_length() - 127 * np.sqrt(3)) < 0.001

  
def test_find_border_targets():
  labels = np.zeros( (257, 257), dtype=np.uint8)
  labels[1:-1,1:-1] = 1 

  dt = edt.edt(labels)
  targets = kimimaro.skeletontricks.find_border_targets(
    dt, labels.astype(np.uint32), wx=100, wy=100
  )

  assert len(targets) == 1
  assert targets[1] == (128, 128)

def test_fix_borders_z():
  labels = np.zeros((256, 256, 256), dtype=np.uint8)
  labels[ 64:196, 64:196, : ] = 128

  skels = kimimaro.skeletonize(
    labels,
    teasar_params={
      'const': 250,
      'scale': 10,
      'pdrf_exponent': 4,
      'pdrf_scale': 100000,
    }, 
    anisotropy=(1,1,1),
    object_ids=None, 
    dust_threshold=1000, 
    progress=True, 
    fix_branching=True, 
    in_place=False, 
    fix_borders=True
  )

  skel = skels[128]

  assert np.all(skel.vertices[:,0] == 129)
  assert np.all(skel.vertices[:,1] == 129)
  assert np.all(skel.vertices[:,2] == np.arange(256))

def test_fix_borders_x():
  labels = np.zeros((256, 256, 256), dtype=np.uint8)
  labels[ :, 64:196, 64:196 ] = 128

  skels = kimimaro.skeletonize(
    labels,
    teasar_params={
      'const': 250,
      'scale': 10,
      'pdrf_exponent': 4,
      'pdrf_scale': 100000,
    }, 
    anisotropy=(1,1,1),
    object_ids=None, 
    dust_threshold=1000, 
    progress=True, 
    fix_branching=True, 
    in_place=False, 
    fix_borders=True
  )

  skel = skels[128]

  assert np.all(skel.vertices[:,0] == np.arange(256))
  assert np.all(skel.vertices[:,1] == 129)
  assert np.all(skel.vertices[:,2] == 129)

def test_fix_borders_y():
  labels = np.zeros((256, 256, 256), dtype=np.uint8)
  labels[ 64:196, :, 64:196 ] = 128

  skels = kimimaro.skeletonize(
    labels,
    teasar_params={
      'const': 250,
      'scale': 10,
      'pdrf_exponent': 4,
      'pdrf_scale': 100000,
    }, 
    anisotropy=(1,1,1),
    object_ids=None, 
    dust_threshold=1000, 
    progress=True, 
    fix_branching=True, 
    in_place=False, 
    fix_borders=True
  )

  skel = skels[128]

  assert np.all(skel.vertices[:,0] == 129)
  assert np.all(skel.vertices[:,1] == np.arange(256))
  assert np.all(skel.vertices[:,2] == 129)

def test_extra_targets():
  labels = np.zeros((256, 256, 1), dtype=np.uint8)
  labels[ 64:196, 64:196, : ] = 128

  def skeletonize(labels, **kwargs):
    return kimimaro.skeletonize(
      labels,
      teasar_params={
        'const': 250,
        'scale': 10,
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
      }, 
      anisotropy=(1,1,1),
      object_ids=None, 
      dust_threshold=1000, 
      progress=True, 
      fix_branching=True, 
      in_place=False, 
      fix_borders=True,
      **kwargs
    )[128]

  skel1 = skeletonize(labels)
  skel2 = skeletonize(labels, extra_targets_after=[ (65, 65, 0) ])

  assert skel1.vertices.size < skel2.vertices.size

  skel3 = skeletonize(labels, extra_targets_before=[ (65, 65, 0) ])

  assert skel3.vertices.size < skel2.vertices.size


def test_parallel():
  labels = np.zeros((256, 256, 128), dtype=np.uint8)
  labels[ 0:128, 0:128, : ] = 1
  labels[ 0:128, 128:256, : ] = 2
  labels[ 128:256, 0:128, : ] = 3
  labels[ 128:256, 128:256, : ] = 4

  skels = kimimaro.skeletonize(
    labels,
    teasar_params={
      'const': 250,
      'scale': 10,
      'pdrf_exponent': 4,
      'pdrf_scale': 100000,
    }, 
    anisotropy=(1,1,1),
    object_ids=None, 
    dust_threshold=1000, 
    progress=True, 
    fix_branching=True, 
    in_place=False, 
    fix_borders=True,
    parallel=2,
  )

  assert len(skels) == 4

def test_dimensions():
  labels = np.zeros((10,), dtype=np.uint8)
  skel = kimimaro.skeletonize(labels)

  labels = np.zeros((10,10), dtype=np.uint8)
  skel = kimimaro.skeletonize(labels)

  labels = np.zeros((10,10,10), dtype=np.uint8)
  skel = kimimaro.skeletonize(labels)

  labels = np.zeros((10,10,10,1), dtype=np.uint8)
  skel = kimimaro.skeletonize(labels)

  try:
    labels = np.zeros((10,10,10,2), dtype=np.uint8)
    skel = kimimaro.skeletonize(labels)
    assert False
  except kimimaro.DimensionError:
    pass

@pytest.mark.parametrize('axis', ('x','y'))
def test_joinability(axis):
  def skeletionize(labels, fix_borders):
    return kimimaro.skeletonize(
      labels,
      teasar_params={
        'const': 10,
        'scale': 10,
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
      }, 
      anisotropy=(1,1,1),
      object_ids=None, 
      dust_threshold=0, 
      progress=True, 
      fix_branching=True, 
      in_place=False, 
      fix_borders=fix_borders,
      parallel=1,
    )

  labels = np.zeros((256, 256, 20), dtype=np.uint8)

  if axis == 'x':
    lslice = np.s_[ 32:160, :, : ]
  elif axis == 'y':
    lslice = np.s_[ :, 32:160, : ]

  labels = np.zeros((256, 256, 20), dtype=np.uint8)
  labels[lslice] = 1

  skels1 = skeletionize(labels[:,:,:10], True)
  skels1 = skels1[1]

  skels2 = skeletionize(labels[:,:,9:], True)
  skels2 = skels2[1]
  skels2.vertices[:,2] += 9

  skels_fb = skels1.merge(skels2)
  assert len(skels_fb.components()) == 1

  skels1 = skeletionize(labels[:,:,:10], False)
  skels1 = skels1[1]

  skels2 = skeletionize(labels[:,:,9:], False)
  skels2 = skels2[1]
  skels2.vertices[:,2] += 9

  skels = skels1.merge(skels2)
  # Ususally this results in 2 connected components,
  # but random variation in how fp is handled can 
  # result in a merge near the tails.
  assert not Skeleton.equivalent(skels, skels_fb)

def test_find_cycle():
  edges = np.array([
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [2, 4]
  ], dtype=np.int32)

  cycle = kimimaro.skeletontricks.find_cycle(edges)

  assert np.all(cycle == np.array([0, 2, 1, 0]))

  edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4], [4, 10], [10, 11], [11, 12], [12, 2],
    [4, 5],
    [5, 6],
    [6, 7],
  ], dtype=np.int32)

  cycle = kimimaro.skeletontricks.find_cycle(edges)
  
  assert np.all(cycle == np.array([
    2, 12, 11, 10, 4, 3, 2
  ]))

  # two loops
  edges = np.array([
    [0, 1], [0, 20], [20, 21], [21, 22], [22, 23], [23, 21],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7], [7, 10], [10, 11], [11, 6]
  ], dtype=np.int32)

  cycle = kimimaro.skeletontricks.find_cycle(edges)
  
  assert np.all(cycle == np.array([
    21, 23, 22, 21
  ])) or np.all(cycle == np.array([ 
    6, 11, 10, 7, 6 
  ]))


def test_join_close_components_simple():
  skel = Skeleton([ 
      (0,0,0), (1,0,0), (10,0,0), (11, 0, 0)
    ], 
    edges=[ (0,1), (2,3) ],
    radii=[ 0, 1, 2, 3 ],
    vertex_types=[ 0, 1, 2, 3 ],
    segid=1337,
  )

  assert len(skel.components()) == 2

  res = kimimaro.join_close_components(skel, radius=np.inf)
  assert len(res.components()) == 1

  res = kimimaro.join_close_components(skel, radius=9)
  assert len(res.components()) == 1
  assert np.all(res.edges == [[0,1], [1,2], [2,3]])

  res = kimimaro.join_close_components(skel, radius=8.5)
  assert len(res.components()) == 2

def test_join_close_components_complex():
  skel = Skeleton([ 
      (0,0,0), (1,0,0),    (4,0,0), (6,0,0),        (20,0,0), (21, 0, 0),
      

      (0,0,5), 
      (0,0,10),
    ], 
    edges=[ (0,1), (2,3), (4,5), (6,7) ],
  )

  assert len(skel.components()) == 4

  res = kimimaro.join_close_components(skel, radius=np.inf)
  assert len(res.components()) == 1

  assert np.all(res.edges == [[0,1], [0,3], [1,2], [3,4], [4,5], [5,6], [6,7]])

def test_join_close_components_by_radius():
  skel = Skeleton([ 
      (0,0,0), (1,0,0), (5,0,0), (11, 0, 0)
    ], 
    edges=[ (0,1), (2,3) ],
    radii=[ 100, 100, 100, 100 ],
    vertex_types=[ 0, 1, 2, 3 ],
    segid=1337,
  )

  res = kimimaro.join_close_components(skel, restrict_by_radius=False)
  assert len(res.components()) == 1
  assert np.all(res.edges == [[0,1], [1,2], [2,3]])

  res = kimimaro.join_close_components(skel, restrict_by_radius=True)
  assert len(res.components()) == 1
  assert np.all(res.edges == [[0,1], [1,2], [2,3]])

  skel.radii = np.array([1,1,1,1], dtype=np.float32)
  res = kimimaro.join_close_components(skel, restrict_by_radius=True)
  assert len(res.components()) == 2
  assert np.all(res.edges == [[0,1], [2,3]])

  skel.radii = np.array([1,0.9,3,1], dtype=np.float32)
  res = kimimaro.join_close_components(skel, restrict_by_radius=True)
  assert len(res.components()) == 2
  assert np.all(res.edges == [[0,1], [2,3]])

  skel.radii = np.array([1,1,3,1], dtype=np.float32)
  res = kimimaro.join_close_components(skel, restrict_by_radius=True)
  assert len(res.components()) == 1
  assert np.all(res.edges == [[0,1], [1,2], [2,3]])


def test_fill_all_holes():
  labels = np.zeros((64, 32, 32), dtype=np.uint32)

  labels[0:32,:,:] = 1
  labels[32:64,:,:] = 8

  noise = np.random.randint(low=1, high=8, size=(30, 30, 30))
  labels[1:31,1:31,1:31] = noise

  noise = np.random.randint(low=8, high=11, size=(30, 30, 30))
  labels[33:63,1:31,1:31] = noise

  noise_labels = np.unique(labels)
  assert set(noise_labels) == set([1,2,3,4,5,6,7,8,9,10])

  result = kimimaro.intake.fill_all_holes(labels)

  filled_labels = np.unique(result)
  assert set(filled_labels) == set([1,8])

def test_fix_avocados():
  labels = np.zeros((256, 256, 256), dtype=np.uint32)

  # fake clipped avocado
  labels[:50, :40, :30] = 1 
  labels[:25, :20, :25] = 2

  # double avocado
  labels[50:100, 40:100, 30:80] = 3
  labels[60:90, 50:90, 40:70] = 4
  labels[60:70, 51:89, 41:69] = 5

  # not an avocado
  labels[200:,200:,200:] = 6 # not a pit
  labels[150:200,200:,200:] = 7 # not a fruit

  fn = lambda lbls: edt.edt(lbls)
  dt = fn(labels)

  labels, dbf, remapping = kimimaro.intake.engage_avocado_protection(
    labels, dt, { 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7 },
    soma_detection_threshold=1, 
    edtfn=fn, 
    progress=True
  )

  uniq = set(np.unique(labels))
  assert uniq == set([0, 1, 2, 3, 4]) # 0,2,5 renumbered
  assert np.all(labels[:50, :40, :30] == 1)
  assert np.all(labels[50:100, 40:100, 30:80] == 2)
  assert np.all(labels[150:200,200:,200:] == 3)
  assert np.all(labels[200:,200:,200:] == 4)


def test_cross_sectional_area():
  labels = np.ones((100,3,3), dtype=bool, order="F")

  vertices = np.array([
    [x,1,1] for x in range(labels.shape[0])
  ])

  edges = np.array([
    [x,x+1] for x in range(labels.shape[0] - 1)
  ])

  skel = Skeleton(vertices, edges, segid=1)
  skel = kimimaro.cross_sectional_area(labels, skel, smoothing_window=5)

  assert len(skel.cross_sectional_area == 100)
  assert np.all(skel.cross_sectional_area == 9)


def test_moving_average():

  data = np.array([])
  assert np.all(moving_average(data, 1) == data)
  assert np.all(moving_average(data, 2) == data)

  data = np.array([1,1,1,1,1,1,1,1,1,1,1])
  assert np.all(moving_average(data, 1) == data)

  data = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
  assert np.all(moving_average(data, 1) == data)

  data = np.array([1,1,1,1,1,10,1,1,1,1,1])
  assert np.all(moving_average(data, 1) == data)

  data = np.array([1,1,1,1,1,1,1,1,1,1,1])
  assert np.all(moving_average(data, 2) == data)

  data = np.array([0,1,1,1,1,1,1,1,1,1,0])
  ans = np.array([
    0,0.5,1,1,1,1,1,1,1,1,0.5
  ])
  assert np.all(moving_average(data, 2) == ans)

  data = np.array([0,1,1,1,1,1,1,1,1,1,0])
  ans = np.array([
    1/3,1/3,2/3,1,1,1,1,1,1,1,2/3
  ])
  res = moving_average(data, 3)
  assert np.all(res == ans)
  assert len(ans) == len(data)

def test_no_fix_branching(connectomics_data):
  kimimaro.skeletonize(connectomics_data[:,:,100], fix_branching=False)


def test_remove_row():
  arr = np.array([
    [0,1],
    [1,2],
    [2,1],
    [2,2],
    [2,3],
    [3,4],
  ])

  result = kimimaro.post.remove_row(arr, np.array([[1,2]]))

  assert np.all(result == np.array([[0,1],[2,2],[2,3],[3,4]]))

  arr = np.array([
    []
  ])

  result = kimimaro.post.remove_row(arr, np.array([[1,2]]))

  assert np.all(result == np.array([]))

def test_cross_sectional_area():
  labels = np.ones([100,100,100], dtype=np.uint8)
  skel = kimimaro.skeletonize(labels, teasar_params={
    "pdrf_exponent": 16,

  })[1]

  xsa_1 = cross_sectional_area(labels, skel, step=1).cross_sectional_area
  xsa_10 = cross_sectional_area(labels, skel, step=10).cross_sectional_area

  assert np.all(xsa_1[xsa_10 == 0] != xsa_10[xsa_10 == 0])
  assert np.all(xsa_1[xsa_10 > 0] == xsa_10[xsa_10 > 0])
  assert np.any(xsa_1 == 10000)

  terminals = skel.terminals()
  assert np.all(xsa_10[terminals] > 0)
  assert np.all(xsa_10[terminals] > 0)

  try:
    cross_sectional_area(labels, skel, step=-1)
  except AssertionError:
    pass
  
def test_postprocess():
  skel = Skeleton([ 
      (0,0,0), (1,0,0),    (4,0,0), (6,0,0),        (20,0,0), (21, 0, 0),
      

      (0,0,5), 
      (0,0,10),
    ], 
    edges=[ (0,1), (2,3), (4,5), (6,7), (0,7), (1,6) ],
  )

  res_skel = kimimaro.post.postprocess(skel, dust_threshold=0, tick_threshold=0)

  ans = Skeleton([ 
      (4,0,0), (6,0,0),        (20,0,0), (21, 0, 0),
    ], 
    edges=[ (0,1), (2,3) ],
  )

  assert Skeleton.equivalent(res_skel, ans)




