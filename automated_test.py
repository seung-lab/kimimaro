import pytest

import edt
import numpy as np
from cloudvolume import *

import kimimaro


def test_binary_image():
  labels = np.ones( (256, 256, 3), dtype=np.bool)
  labels[-1,0] = 0
  labels[0,-1] = 0
  
  skels = kimimaro.skeletonize(labels, fix_borders=False)

  assert len(skels) == 1


def test_square():
  labels = np.ones( (1000, 1000), dtype=np.uint8)
  labels[-1,0] = 0
  labels[0,-1] = 0
  
  skels = kimimaro.skeletonize(labels, fix_borders=False)

  assert len(skels) == 1

  skel = skels[1]
  assert skel.vertices.shape[0] == 1000
  assert skel.edges.shape[0] == 999
  assert abs(skel.cable_length() - 999 * np.sqrt(2)) < 0.001

  labels = np.ones( (1000, 1000), dtype=np.uint8)
  labels[0,0] = 0
  labels[-1,-1] = 0

  skels = kimimaro.skeletonize(labels, fix_borders=False)

  assert len(skels) == 1

  skel = skels[1]
  assert skel.vertices.shape[0] == 1000
  assert skel.edges.shape[0] == 999
  assert abs(skel.cable_length() - 999 * np.sqrt(2)) < 0.001

def test_cube():
  labels = np.ones( (256, 256, 256), dtype=np.uint8)
  labels[0, 0, 0] = 0
  labels[-1, -1, -1] = 0
  
  skels = kimimaro.skeletonize(labels, fix_borders=False)

  assert len(skels) == 1

  skel = skels[1]
  assert skel.vertices.shape[0] == 256
  assert skel.edges.shape[0] == 255
  assert abs(skel.cable_length() - 255 * np.sqrt(3)) < 0.001

  
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
    cc_safety_factor=1,
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
    cc_safety_factor=1,
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
    cc_safety_factor=1,
    progress=True, 
    fix_branching=True, 
    in_place=False, 
    fix_borders=True
  )

  skel = skels[128]

  assert np.all(skel.vertices[:,0] == 129)
  assert np.all(skel.vertices[:,1] == np.arange(256))
  assert np.all(skel.vertices[:,2] == 129)

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
    cc_safety_factor=1,
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

def test_joinability():
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
      cc_safety_factor=1,
      progress=True, 
      fix_branching=True, 
      in_place=False, 
      fix_borders=fix_borders,
      parallel=1,
    )

  def testlabels(labels):
    skels1 = skeletionize(labels[:,:,:10], True)
    skels1 = skels1[1]

    skels2 = skeletionize(labels[:,:,9:], True)
    skels2 = skels2[1]
    skels2.vertices[:,2] += 9

    skels = skels1.merge(skels2)
    assert len(skels.components()) == 1

    skels1 = skeletionize(labels[:,:,:10], False)
    skels1 = skels1[1]

    skels2 = skeletionize(labels[:,:,9:], False)
    skels2 = skels2[1]
    skels2.vertices[:,2] += 9

    skels = skels1.merge(skels2)
    assert len(skels.components()) == 2

  labels = np.zeros((256, 256, 20), dtype=np.uint8)
  labels[ :, 32:160, : ] = 1
  testlabels(labels)

  labels = np.zeros((256, 256, 20), dtype=np.uint8)
  labels[ 32:160, :, : ] = 1
  testlabels(labels)
