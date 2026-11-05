"""
Tests of kimimaro's GPU backend on real hardware (the brook-cu12 wheel and a CUDA
GPU): the output contract, agreement with the CPU backend, and that every skeletonize
option reaches Brook. Tests marked gpu skip without brook or a visible CUDA device; an
installed but unsupported GPU or driver fails them instead. The soma tests also run on the CPU.
"""
import copy
import os
from functools import partial

import cc3d
import edt
import fastremap
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.csgraph
from osteoid import Skeleton
from scipy.spatial import cKDTree

import kimimaro
from kimimaro.intake import DEFAULT_TEASAR_PARAMS as DEFAULT_PARAMS

from automated_test_gpu import REPO_ROOT, real_brook, run_python

@pytest.fixture
def gpu():
  real_brook()
  from brook.device import device_count, require_gpu
  # Skip only when no device is visible. An unsupported driver or GPU must
  # fail the test with Brook's startup-check message, not skip it.
  if not device_count():
    pytest.skip("no visible CUDA device")
  require_gpu()

# --- synthetic volumes (deterministic; voxel space is isotropic,
# anisotropy only enters through the skeletonize arguments) ---

def _capsule(vol, label, a, b, radius):
  """Paint the capsule around the voxel-space segment a-b."""
  a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
  lo = np.maximum(np.floor(np.minimum(a, b) - radius - 1), 0).astype(int)
  hi = np.minimum(np.ceil(np.maximum(a, b) + radius + 2), vol.shape).astype(int)
  grid = np.stack(np.meshgrid(*[np.arange(l, h, dtype=np.float64) for l, h in zip(lo, hi)], indexing="ij"), axis=-1)
  ab = b - a
  t = np.clip(((grid - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
  dist = np.linalg.norm(grid - (a + t[..., None] * ab), axis=-1)
  vol[tuple(slice(l, h) for l, h in zip(lo, hi))][dist <= radius] = label

def _tube(vol, label, points, radius, rng):
  # Axis and caliber wobble a little (seeded), like a segmented neurite: perfectly
  # smooth tubes have plateaus in the distance field where Brook and Kimimaro
  # legitimately break ties differently.
  pts = [np.asarray(p, dtype=np.float64) for p in points]
  for a, b in zip(pts[:-1], pts[1:]):
    n = max(int(np.ceil(np.linalg.norm(b - a) / 3)), 1)
    knots = [a + (b - a) * i / n + rng.normal(0, 0.6, 3) * (0 < i < n) for i in range(n + 1)]
    for p, q in zip(knots[:-1], knots[1:]):
      _capsule(vol, label, p, q, radius * rng.uniform(0.8, 1.2))

def _vol_neuron(soma=9):
  """A cell body of radius `soma` with five curved neurites of different caliber."""
  rng, vol = np.random.RandomState(1), np.zeros((80, 80, 80), dtype=np.uint32)
  _capsule(vol, 1, (40, 40, 40), (40, 40, 40), soma) # a ball
  _tube(vol, 1, [(40, 40, 40), (62, 46, 38), (76, 44, 50)], 2.5, rng)
  _tube(vol, 1, [(40, 40, 40), (22, 30, 40), (6, 36, 30)], 3, rng)
  _tube(vol, 1, [(40, 40, 40), (44, 62, 44), (36, 74, 60)], 2, rng)
  _tube(vol, 1, [(40, 40, 40), (38, 20, 36), (46, 8, 24)], 2, rng)
  _tube(vol, 1, [(40, 40, 40), (42, 40, 62), (30, 46, 76)], 2.5, rng)
  return vol

def _grow(vol, rng, start, direction, radius, length, depth):
  end = np.asarray(start) + length * direction
  _tube(vol, 3, [start, end], radius, rng)
  if depth == 0 or radius < 1.5:
    return
  for _ in range(2):
    turn = rng.normal(size=3)
    new = direction + 0.9 * turn / np.linalg.norm(turn)
    new /= np.linalg.norm(new)
    _grow(vol, rng, end, new, radius * 0.8, length * 0.8, depth - 1)

def _vol_tree(): # a trunk that splits three times
  vol = np.zeros((96, 64, 64), dtype=np.uint32)
  _grow(vol, np.random.RandomState(7), (6, 32, 32), np.array([1.0, 0, 0]), 3.5, 22, 3)
  return vol

def _vol_multi(): # four objects with distinct, unordered labels; 7 touches the volume border
  rng, vol = np.random.RandomState(2), np.zeros((72, 72, 72), dtype=np.uint32)
  _tube(vol, 7, [(0, 10, 10), (30, 12, 12), (40, 30, 20)], 4, rng)
  _tube(vol, 41, [(10, 30, 50), (40, 36, 54), (64, 60, 50)], 3, rng)
  _tube(vol, 1000, [(14, 62, 8), (30, 50, 12), (50, 66, 8)], 2.5, rng)
  _tube(vol, 123456, [(20, 20, 40), (30, 30, 66), (60, 20, 62)], 3.5, rng)
  return vol

def _vol_bar(axis):
  """A square bar of odd side (so its axis is unambiguous) spanning the volume along `axis`."""
  vol = np.zeros((40, 40, 40), dtype=np.uint8)
  vol[tuple(slice(None) if a == axis else slice(10, 31) for a in range(3))] = 128
  return vol

# benchmarks/benchmark.py: the defaults, but for the soma detection and invalidation
BENCHMARK_PARAMS = dict(DEFAULT_PARAMS, soma_detection_threshold=1100, soma_invalidation_scale=1.0)

def _kw(anisotropy, **kwargs):
  return dict(dict(teasar_params=DEFAULT_PARAMS, dust_threshold=100), **kwargs, anisotropy=anisotropy, progress=False)

# kimimaro's test_fix_borders_*: a path this stiff leaves one straight skeleton
# with fix_borders=True and a different, longer one without
BAR_KW = _kw((40, 32, 20), teasar_params=dict(const=250, scale=10, pdrf_exponent=4, pdrf_scale=100000))

CASES = {
  "neuron": (_vol_neuron, _kw((16, 16, 40))),
  "tree": (_vol_tree, _kw((4, 4, 10), teasar_params=dict(DEFAULT_PARAMS, const=12))),
  "multi": (_vol_multi, _kw((16, 16, 40))),
  # a cell body whose dbf is 20 * 200 = 4000 > soma_acceptance_threshold (3500)
  "soma": (partial(_vol_neuron, soma=20), _kw((200, 200, 200), teasar_params=BENCHMARK_PARAMS)),
  "neuron_2d": (lambda: _vol_neuron()[:, :, 40].copy(), _kw((16, 16, 40))),
  "empty": (lambda: np.zeros((20, 20, 20), dtype=np.uint8), _kw((1, 1, 1), dust_threshold=0)),
  **{"bar%d" % axis: (partial(_vol_bar, axis), BAR_KW) for axis in range(3)},
}

def _case(name):
  build, kw = CASES[name]
  return build(), kw

def _run(vol, kwargs, on_gpu, **overrides):
  return kimimaro.skeletonize(vol.copy(order="K"), gpu=on_gpu, **dict(kwargs, **overrides))

@pytest.fixture(scope="module")
def real_crop():
  """A 128^3 window of benchmarks/connectomics.npy.ckl.gz (~140 objects) and its CPU skeletons."""
  crackle = pytest.importorskip("crackle")
  path = os.path.join(REPO_ROOT, "benchmarks", "connectomics.npy.ckl.gz")
  if not os.path.exists(path):
    pytest.skip("benchmarks/connectomics.npy.ckl.gz is not available")
  vol = np.asfortranarray(crackle.load(path)[0:128, 384:512, 192:320])
  kw = _kw((16, 16, 40), teasar_params=BENCHMARK_PARAMS, dust_threshold=1000)
  return vol, kw, _run(vol, kw, False)

@pytest.fixture
def case(request, gpu): # (name, volume, kwargs, CPU skeletons) of the case named by the indirect parameter
  if request.param == "real":
    return (request.param,) + request.getfixturevalue("real_crop") # after `gpu`: no CPU work if the test skips
  vol, kw = _case(request.param)
  return request.param, vol, kw, _run(vol, kw, False)

def _n_components(skel):
  n = len(skel.vertices)
  graph = scipy.sparse.coo_matrix((np.ones(len(skel.edges)), tuple(skel.edges.T)), shape=(n, n))
  return int(scipy.sparse.csgraph.connected_components(graph, directed=False)[0])

def _n_endpoints(skel):
  return int(np.count_nonzero(np.bincount(skel.edges.ravel(), minlength=len(skel.vertices)) == 1))

def _voxels(skel, anisotropy):
  return skel.vertices.astype(np.float64) / np.asarray(anisotropy, dtype=np.float64)

def _parity(cpu, gpu, anisotropy):
  """
  Metrics of two {label: Skeleton} dicts with the same labels: relative difference of cable
  length, nearest-vertex distance in voxels (symmetric, pooled and mean per object), GPU/CPU
  ratio of end points (degree 1), and connected components per object. Total and worst object.
  """
  labels = sorted(cpu)
  nn, cable_c, cable_g, ends_c, ends_g = [], [], [], [], []
  for k in labels:
    cv, gv = _voxels(cpu[k], anisotropy), _voxels(gpu[k], anisotropy)
    nn.append(np.concatenate([cKDTree(gv).query(cv)[0], cKDTree(cv).query(gv)[0]]))
    cable_c.append(float(cpu[k].cable_length()))
    cable_g.append(float(gpu[k].cable_length()))
    ends_c.append(_n_endpoints(cpu[k]))
    ends_g.append(_n_endpoints(gpu[k]))
  pooled, cable_c, cable_g, ends_c, ends_g = np.concatenate(nn), *(np.array(a) for a in (cable_c, cable_g, ends_c, ends_g))
  return dict(
    cable_total_rel=(cable_g.sum() - cable_c.sum()) / cable_c.sum(),
    cable_obj_rel=float(np.max(np.abs(cable_g - cable_c) / cable_c)),
    nn_mean=float(pooled.mean()),
    nn_p95=float(np.percentile(pooled, 95)),
    nn_mean_obj=float(max(d.mean() for d in nn)),
    endpoint_ratio=ends_g.sum() / ends_c.sum() if ends_c.sum() else float(ends_g.sum() == 0),
    endpoint_diff_obj=int(np.max(np.abs(ends_g - ends_c))),
    cpu_components={k: _n_components(cpu[k]) for k in labels},
    gpu_components={k: _n_components(gpu[k]) for k in labels},
  )

# Thresholds on the CPU/GPU agreement, set from the worst values observed for
# these cases (every comparison made below, the same cases with `const`
# varied by -20 % ... +25 %, and a 256^3 crop with 478 objects), plus margin.
# Where the two backends take different but equally cheap routes (Brook
# goes straight along an axis, Kimimaro alternates between slices) the cable
# differs by a few percent and a single object can differ by more, so the
# per-object limits are looser still.
PARITY_CABLE_TOTAL = 0.10     # worst seen 0.064 (neuron, fix_branching=False)
PARITY_NN_MEAN = 0.5          # worst seen 0.30 (same case)
PARITY_NN_P95 = 2.0           # worst seen 1.41 (diagonal neighbours); allows one step more
PARITY_ENDPOINT_RATIO = (0.8, 1.25)  # worst seen 0.964 .. 1.016; one end point in five
PARITY_CABLE_OBJECT = 0.25    # worst seen 0.122 (256^3 crop), 0.214 (real crop, const x0.9)
PARITY_NN_MEAN_OBJECT = 2.0   # worst seen 1.11 (256^3 crop), 1.42 (real crop, const x0.9)
PARITY_NN_MEAN_OBJECT_FEW = 1.0  # for volumes of <= 10 objects: worst seen 0.73 (multi)
PARITY_ENDPOINT_DIFF_OBJECT = 3  # worst seen 3 (real crop, const x0.9), 2 at 256^3; no margin on the const-sweep worst

def _assert_parity(cpu, gpu, anisotropy, per_object=True):
  assert set(cpu) == set(gpu), (sorted(cpu), sorted(gpu))
  if not cpu:
    return
  m = _parity(cpu, gpu, anisotropy)
  limits = dict(cable_total_rel=PARITY_CABLE_TOTAL, nn_mean=PARITY_NN_MEAN, nn_p95=PARITY_NN_P95)
  if per_object:
    limits.update(cable_obj_rel=PARITY_CABLE_OBJECT, endpoint_diff_obj=PARITY_ENDPOINT_DIFF_OBJECT,
      nn_mean_obj=PARITY_NN_MEAN_OBJECT_FEW if len(cpu) <= 10 else PARITY_NN_MEAN_OBJECT)
  for name, limit in limits.items():
    assert abs(m[name]) <= limit, (name, m)
  assert PARITY_ENDPOINT_RATIO[0] <= m["endpoint_ratio"] <= PARITY_ENDPOINT_RATIO[1], m
  for label, count in m["gpu_components"].items():
    # Brook may join what Kimimaro leaves apart (soma), never the opposite
    assert count <= m["cpu_components"][label], (label, m)

def _assert_contract(skels, volume, anisotropy, components=1, skip=()):
  """
  What kimimaro.skeletonize promises about its result. `components=None` allows several pieces
  per label (real data). `skip` exempts a case from "adjacent" (every edge joins 26-adjacent
  voxels; soma skeletons have long straight edges by design, so do the pieces postprocess joins),
  "radii" (a radius is the distance field at its vertex; voxel_graph changes the field) or
  "frame" (space and transform; postprocess resets them on a skeleton it had to join).
  """
  volume = volume.reshape(volume.shape + (1,) * (3 - volume.ndim))
  aniso = np.asarray(anisotropy, dtype=np.float64)
  assert type(skels) is dict and set(skels) <= set(np.unique(volume).tolist()) - {0}
  if skels and "radii" not in skip:
    labels = fastremap.renumber(volume, in_place=False)[0]
    field = edt.edt(labels, anisotropy=tuple(anisotropy), black_border=bool(labels.min() == labels.max()))

  for key, skel in skels.items():
    n = len(skel.vertices)
    assert type(key) is int and isinstance(skel, Skeleton) and skel.id == key
    assert skel.vertices.dtype == np.float32 and skel.vertices.shape == (n, 3)
    assert skel.edges.dtype == np.uint32 and skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.radii.dtype == np.float32 and skel.radii.shape == (n,) and np.all(skel.radii > 0)
    if "frame" not in skip:
      assert skel.space == "physical" and np.array_equal(skel.transform, np.diag(np.append(aniso, 1))[:3].astype(np.float32))
    assert len(skel.edges) >= 1 and skel.edges.max() < n
    assert len(np.unique(np.sort(skel.edges, axis=1), axis=0)) == len(skel.edges)
    voxels = skel.vertices.astype(np.float64) / aniso
    index = np.rint(voxels).astype(np.int64)
    assert np.abs(voxels - index).max() < 1e-3 # vertices are voxel centers
    assert np.all(index >= 0) and np.all(index < volume.shape)
    assert np.all(volume[tuple(index.T)] == key) # on their own label
    if "adjacent" not in skip:
      # a skeleton that skips vertices (or is shifted onto other ones) would
      # still be close to the CPU's, so the parity limits alone cannot see it
      steps = np.abs(index[skel.edges[:, 0]] - index[skel.edges[:, 1]])
      assert steps.max() <= 1, (key, steps.max())
    if "radii" not in skip:
      assert np.allclose(skel.radii, field[tuple(index.T)], rtol=1e-4, atol=0)
    count = _n_components(skel)
    assert components is None or count == components
    assert len(skel.edges) == n - count # a forest

def _same(a, b):
  """Bit-identical keys, vertices, edges and radii."""
  return list(a) == list(b) and all(
    getattr(a[k], f).tobytes() == getattr(b[k], f).tobytes() for k in a for f in ("vertices", "edges", "radii")
  )

def _compare(vol, kwargs, keys=None, parity=True, contract_vol=None, **contract):
  """Both backends, same arguments: same labels (`keys`, if given), the GPU result obeys the contract
  (of `contract_vol`, if the options change the labels) and, if `parity`, agrees with the CPU's."""
  cpu, out = _run(vol, kwargs, False), _run(vol, kwargs, True)
  assert set(out) == set(cpu) and (keys is None or set(cpu) == set(keys))
  _assert_contract(out, vol if contract_vol is None else contract_vol, kwargs["anisotropy"], **contract)
  if parity:
    _assert_parity(cpu, out, kwargs["anisotropy"])
  return cpu, out

@pytest.mark.gpu
@pytest.mark.parametrize("case", ("neuron", "tree", "multi", "soma", "real"), indirect=True)
def test_gpu_result_obeys_the_contract_and_agrees_with_cpu(case):
  name, vol, kw, cpu = case
  out = _run(vol, kw, True)
  # (the soma skeleton joins the cell body's center to the surface with long straight edges)
  _assert_contract(out, vol, kw["anisotropy"], components=None if name == "real" else 1, skip=("adjacent",) if name == "soma" else ())
  _assert_parity(cpu, out, kw["anisotropy"])

TREE_PARAMS = CASES["tree"][1]["teasar_params"]
OPTIONS = {
  # name: (case, changed skeletonize arguments, the labels both backends return). The parity limits are calibrated
  # for whole volumes: a few pooled objects of `multi` that include 123456 (the one whose route differs) sit on them.
  "object_ids": ("multi", dict(object_ids=[41]), {41}),
  "object_ids_one_absent": ("multi", dict(object_ids=[7, 41, 99999]), {7, 41}),
  "object_ids_numpy_ints": ("multi", dict(object_ids=[np.uint32(41), np.uint32(1000)]), {41, 1000}),
  "object_ids_all_absent": ("multi", dict(object_ids=[99999]), set()),
  "dust_threshold": ("multi", dict(dust_threshold=3000), {7}), # 41, 1000 and 123456 have 2356, 1112 and 2893 voxels
  "non_integer_anisotropy": ("tree", dict(anisotropy=(3.3, 7.7, 11.1), teasar_params=dict(TREE_PARAMS, const=88)), {3}),
  "max_paths_2": ("tree", dict(teasar_params=dict(TREE_PARAMS, max_paths=2)), {3}),
  "max_paths_1": ("tree", dict(teasar_params=dict(TREE_PARAMS, max_paths=1)), set()), # too few for the tree: given up
  "two_dimensional_input": ("neuron_2d", {}, {1}),
  "empty_volume": ("empty", {}, set()),
}

@pytest.mark.gpu
@pytest.mark.parametrize("name,changes,keys", OPTIONS.values(), ids=list(OPTIONS))
def test_option_matches_cpu(gpu, name, changes, keys):
  vol, kw = _case(name)
  _compare(vol, dict(kw, **changes), keys)

@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ("uint32", "uint64", "bool"))
def test_label_dtypes(gpu, dtype):
  vol, kw = _case("neuron" if dtype == "bool" else "multi")
  vol = vol.astype(dtype)
  if dtype != "bool": # the largest label the dtype holds (below 2**63 for uint64)
    vol[vol == 123456] = {"uint32": 2**32 - 1, "uint64": 2**63 - 1}[dtype]
  _compare(vol, kw, keys=[1] if dtype == "bool" else None)

@pytest.mark.gpu
@pytest.mark.parametrize("option", ("extra_targets_before", "extra_targets_after"))
def test_extra_targets_are_reached(gpu, option):
  vol, kw = _case("multi")
  targets = {label: tuple(np.argwhere(vol == label)[-1].tolist()) for label in (7, 41, 1000, 123456)} # surface voxels
  cpu, out = _compare(vol, dict(kw, **{option: list(targets.values())}), keys=targets)
  for label, target in targets.items():
    for skel in (cpu[label], out[label]):
      assert np.any(np.all(np.rint(_voxels(skel, kw["anisotropy"])) == target, axis=1))

@pytest.mark.gpu
@pytest.mark.parametrize("flag,name", [("fix_borders", "bar%d" % axis) for axis in range(3)] + [("fix_branching", "tree"), ("fix_branching", "neuron")])
def test_boolean_flags_change_the_result_like_the_cpu(gpu, flag, name):
  vol, kw = _case(name)
  (cpu_on, gpu_on), (cpu_off, gpu_off) = (_compare(vol, dict(kw, **{flag: value})) for value in (True, False))
  geometry = lambda skels: [s.vertices.tobytes() for s in skels.values()]
  # the flag changes the skeleton on the CPU here, so it must on the GPU
  assert geometry(cpu_on) != geometry(cpu_off) and geometry(gpu_on) != geometry(gpu_off)

@pytest.mark.gpu
@pytest.mark.parametrize("option", (None, "fill_holes", "fix_avocados"), ids=("neither", "fill_holes", "fix_avocados"))
def test_fill_holes_and_fix_avocados_absorb_what_they_should(gpu, option):
  vol = _vol_neuron(soma=20)
  _capsule(vol, 2, (40, 40, 40), (40, 40, 40), 9) # a nucleus, segmented separately inside its cell body
  filled = (vol > 0).astype(vol.dtype) # fill_holes: the nucleus and the void below are part of the cell body
  _capsule(vol, 0, (41, 40, 55), (41, 40, 55), 3) # a void on a neurite's path; fix_avocados only absorbs the nucleus
  expected = {None: vol, "fill_holes": filled, "fix_avocados": (vol > 0).astype(vol.dtype)}[option]
  # Without either, the skeleton of the hollow shell wanders over equally good ridge voxels: only the contract is checked.
  _compare(vol, _kw((50, 50, 50), **({option: True} if option else {})), {1} if option else {1, 2}, parity=bool(option), contract_vol=expected)

@pytest.mark.gpu
def test_voxel_graph(gpu):
  vol, kw = _case("neuron")
  split = vol.copy()
  split[30:] *= 2 # a finer segmentation; its graph forbids moving across the plane x = 30
  graph = cc3d.voxel_connectivity_graph(split, connectivity=26)
  _compare(vol, kw, keys=[1])
  _compare(vol, dict(kw, voxel_graph=graph), keys=[1], components=2, skip=("radii",))

@pytest.mark.gpu
@pytest.mark.parametrize("order,extra", (("C", {}), ("F", {"fill_holes": True})))
def test_in_place_false_leaves_the_input_alone(gpu, order, extra):
  vol, kw = _case("multi")
  vol = np.asarray(vol, order=order)
  state = lambda v: (v.dtype, v.shape, v.strides, v.flags.writeable, v.tobytes(order="A"))
  before = state(vol)
  assert kimimaro.skeletonize(vol, gpu=True, in_place=False, **dict(kw, **extra))
  assert state(vol) == before

@pytest.mark.gpu
def test_repeated_runs_and_memory_orders_are_bit_identical(gpu):
  vol, kw = _case("multi")
  first = _run(np.ascontiguousarray(vol), kw, True)
  assert _same(first, _run(np.ascontiguousarray(vol), kw, True))
  assert _same(first, _run(np.asfortranarray(vol), kw, True))

def _postprocessed(skels):
  done = {k: kimimaro.postprocess(copy.deepcopy(s), dust_threshold=500) for k, s in skels.items()}
  return {k: s for k, s in done.items() if not s.empty()}

@pytest.mark.gpu
@pytest.mark.parametrize("case", ("neuron", "multi", "real"), indirect=True)
def test_postprocess_on_gpu_output(case):
  _, vol, kw, cpu = case
  out = _postprocessed(_run(vol, kw, True))
  assert out
  # (postprocess joins the pieces of a label with long edges, hence adjacent; and
  # its thresholded steps can drop a different branch of one object, hence per_object)
  _assert_contract(out, vol, kw["anisotropy"], components=None, skip=("frame", "adjacent"))
  _assert_parity(_postprocessed(cpu), out, kw["anisotropy"], per_object=False)

@pytest.mark.gpu
def test_join_close_components_on_gpu_output(gpu):
  vol, kw = _case("multi")
  skels = list(_run(vol, kw, True).values())
  joined = kimimaro.join_close_components(copy.deepcopy(skels))
  assert len(joined.vertices) == sum(len(s.vertices) for s in skels)
  assert _n_components(joined) == 1

@pytest.mark.gpu
def test_cross_sectional_area_on_gpu_output(gpu):
  vol, kw = _case("multi")
  aniso, gpu_skels, cpu_skels = kw["anisotropy"], _run(vol, kw, True), _run(vol, kw, False)
  _assert_contract(gpu_skels, vol, aniso) # first: xs3d crashes the interpreter on a vertex outside the image
  out, cpu = (kimimaro.cross_sectional_area(vol, copy.deepcopy(s), anisotropy=aniso, progress=False) for s in (gpu_skels, cpu_skels))
  assert set(out) == set(gpu_skels)
  for key, skel in out.items():
    area = skel.cross_sectional_area
    assert area.dtype == np.float32 and area.shape == (len(skel.vertices),) and np.all(area > 0)
    assert area.max() <= np.count_nonzero(vol == key) * float(np.prod(aniso[:2])) # at most the object's own cross section
    assert abs(area.mean() / cpu[key].cross_sectional_area.mean() - 1) < 0.1

# --- kimimaro's own multiprocessing (parallel >= 2 spawns processes that share the
# volume through shared memory) next to the GPU in one process, in a new
# interpreter, where this one's CUDA context cannot hide a problem ---

LIFECYCLE_SCRIPT = """
  import numpy as np, kimimaro
  vol = np.zeros((40, 40, 40), dtype=np.uint8)
  for i in (1, 2, 3):
    vol[4:36, 10 * i - 5:10 * i - 1, 10 * i - 5:10 * i - 1] = i
  for step in ORDER:
    out = kimimaro.skeletonize(vol.copy(), parallel=2 if step == "cpu" else 1, gpu=step == "gpu", dust_threshold=0, progress=False)
    assert sorted(out) == [1, 2, 3], (step, sorted(out))
"""

@pytest.mark.gpu
@pytest.mark.parametrize("order", (("cpu", "gpu"), ("gpu", "cpu")), ids=("cpu_then_gpu", "gpu_then_cpu"))
def test_parallel_cpu_and_gpu_in_one_process(gpu, order):
  run_python("\n  ORDER = {!r}".format(order) + LIFECYCLE_SCRIPT)

# --- soma regressions (small volumes). They assert only what both backends do
# today: the skeleton stays connected and keeps its root. What differs between
# the backends is deliberately not pinned. The CPU variants need no brook and
# run everywhere; the GPU ones are marked. ---

SOMA_PARAMS = dict(BENCHMARK_PARAMS, const=3, soma_detection_threshold=6, soma_acceptance_threshold=8, soma_invalidation_const=3)
C48, C40 = (48, 48, 48), (40, 40, 40)

SOMA_CASES = {
  # name: (volume shape, soma center, soma radius, [(neurite radius, point, point, ...)], teasar params, more skeletonize arguments)
  # three dendrites and a side branch of the +x one
  "side_branch": ((96,) * 3, C48, 12, [
    (2.5, C48, (90, 48, 48)), (2.5, C48, (48, 8, 48)), (2.5, C48, (70, 70, 85)), (2.0, (75, 48, 48), (75, 88, 48))], SOMA_PARAMS, {}),
  # a dendrite leaves the soma and comes back, ending inside the invalidation radius
  "hook": ((96,) * 3, C48, 12, [
    (2.0, C48, (80, 48, 48), (80, 66, 48), (52, 66, 48)), (2.0, C48, (48, 8, 48))], dict(SOMA_PARAMS, soma_invalidation_const=10), {}),
  "lone": ((64,) * 3, (32, 32, 32), 12, [], SOMA_PARAMS, {}), # a ball alone: the soma invalidation (12 + 3) removes every voxel
  # a path vertex lies at exactly the soma radius (sqrt(145)) from the root
  "tie": ((80,) * 3, C40, 12, [(1.0, (40, 41, 40), (76, 41, 40))], dict(SOMA_PARAMS, soma_invalidation_const=0), {}),
  # a soma touching the x=0 face, where fix_borders puts a target inside the soma
  "border": ((80,) * 3, (6, 40, 40), 14, [
    (2.5, (6, 40, 40), (70, 40, 40)), (2.5, (6, 40, 40), (30, 75, 40))], SOMA_PARAMS, {"fix_borders": True}),
}

def _soma_case(name):
  shape, center, radius, neurites, params, extra = SOMA_CASES[name]
  vol = np.zeros(shape, dtype=np.uint32)
  _capsule(vol, 1, center, center, radius)
  for r, *points in neurites:
    for a, b in zip(points[:-1], points[1:]):
      _capsule(vol, 1, a, b, r)
  return vol, _kw((1, 1, 1), teasar_params=params, dust_threshold=0, fix_branching=True, **extra), center

@pytest.fixture(params=("cpu", pytest.param("gpu", marks=pytest.mark.gpu)))
def soma_on_gpu(request):
  if request.param == "gpu":
    request.getfixturevalue("gpu")
  return request.param == "gpu"

@pytest.mark.parametrize("name", ("side_branch", "hook", "tie", "border"))
def test_soma_skeleton_stays_connected_and_keeps_its_root(soma_on_gpu, name):
  vol, kw, root = _soma_case(name)
  skel = _run(vol, kw, soma_on_gpu)[1]
  assert _n_components(skel) == 1
  if name in ("tie", "border"):
    assert len(skel.edges) == len(skel.vertices) - 1 # a tree
  assert np.any(np.all(skel.vertices == root, axis=1))

def test_lone_soma_has_no_edges(soma_on_gpu):
  vol, kw, _ = _soma_case("lone")
  assert all(len(skel.edges) == 0 for skel in _run(vol, kw, soma_on_gpu).values())
