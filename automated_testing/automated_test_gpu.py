"""
GPU-free tests of kimimaro's optional GPU backend. Tier A fakes the `brook`
package in sys.modules (passes with or without brook-cu12). Tier B runs the real
brook when installed with no CUDA device visible (CUDA_VISIBLE_DEVICES=).
Real-GPU tests: automated_test_gpu_hw.py; `forge --gpu`: automated_test_cli.py.
"""
import inspect
import os
import subprocess
import sys
import textwrap
import types
import warnings

import pytest

import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
  import tomllib
except ImportError: # Python < 3.11 (pytest depends on tomli there)
  import tomli as tomllib

import kimimaro
import kimimaro.gpu
import kimimaro.intake

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTALL_HINT_PARTS = (
  'pip install "kimimaro[gpu]"',
  "Linux x86-64",
  "Python 3.12",
  "NVIDIA",
)

# what kimimaro.gpu.skeletonize hands to brook.skeletonize
# (`streaming` is conditional, parallel / parallel_chunk_size never)
BROOK_KEYWORDS = {
  "teasar_params", "anisotropy", "object_ids", "dust_threshold",
  "progress", "fix_branching", "in_place", "fix_borders",
  "extra_targets_before", "extra_targets_after",
  "fill_holes", "fix_avocados", "voxel_graph",
}

# kimimaro's defaults of the boolean options
BOOL_DEFAULTS = dict(
  progress=True, fix_branching=True, in_place=False,
  fix_borders=True, fill_holes=False, fix_avocados=False,
)

class FakeBrook(object):
  """
  Stand-in for the brook package. Behavior is configured by setting
  attributes at any time before the call under test; every call to
  the fake is recorded.
  """
  def __init__(self):
    self.result = {"fake_result": object()} # compared by identity
    self.require_gpu_error = None
    self.device_count_error = None
    self.device_count_value = 1
    self.skeletonize_error = None

    self.calls = []  # (args, kwargs) of each brook.skeletonize call
    self.events = [] # order of calls: "require_gpu", "skeletonize"

    self.device = types.ModuleType("brook.device")
    self.device.require_gpu = self._require_gpu
    self.device.device_count = self._device_count

    self.module = types.ModuleType("brook")
    self.module.__path__ = []
    self.module.device = self.device
    self.module.DimensionError = type("DimensionError", (Exception,), {})
    self.module.skeletonize = self._skeletonize

  @property
  def DimensionError(self):
    return self.module.DimensionError

  def _require_gpu(self, *args, **kwargs):
    self.events.append("require_gpu")
    if self.require_gpu_error is not None:
      raise self.require_gpu_error

  def _device_count(self):
    if self.device_count_error is not None:
      raise self.device_count_error
    return self.device_count_value

  def _skeletonize(self, *args, **kwargs):
    self.events.append("skeletonize")
    self.calls.append((args, kwargs))
    if self.skeletonize_error is not None:
      raise self.skeletonize_error
    return self.result

  def only_call(self):
    assert len(self.calls) == 1
    return self.calls[0]

def install_brook(monkeypatch, fake):
  monkeypatch.setitem(sys.modules, "brook", fake.module)
  monkeypatch.setitem(sys.modules, "brook.device", fake.device)

def block_brook(monkeypatch):
  # a None entry makes `import brook` raise ImportError
  monkeypatch.setitem(sys.modules, "brook", None)
  monkeypatch.setitem(sys.modules, "brook.device", None)

def real_brook():
  """The installed brook package; skips if it is missing or not Brook."""
  brook = pytest.importorskip("brook")
  if not hasattr(brook, "skeletonize"):
    pytest.skip("the installed 'brook' module is not the Brook GPU skeletonizer")
  return brook

def run_python(code, timeout=120):
  proc = subprocess.run(
    [sys.executable, "-c", textwrap.dedent(code)],
    cwd=REPO_ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
    timeout=timeout,
  )
  assert proc.returncode == 0, "exit status {}, stderr:\n{}".format(proc.returncode, proc.stderr)
  return proc.stdout

@pytest.fixture
def fake_brook(monkeypatch):
  fake = FakeBrook()
  install_brook(monkeypatch, fake)
  return fake

@pytest.fixture
def labels():
  return np.ones((4, 4, 4), dtype=np.uint32)

def tube():
  labels = np.zeros((24, 24, 24), dtype=np.uint8)
  labels[4:20, 10:14, 10:14] = 1
  return labels

# --- import isolation: brook is needed only once gpu=True is used
def test_cpu_path_never_imports_brook_or_gpu_module():
  out = run_python("""
    import sys
    import numpy as np
    import kimimaro
    labels = np.zeros((24, 24, 24), dtype=np.uint8)
    labels[4:20, 10:14, 10:14] = 1
    assert len(kimimaro.skeletonize(labels, dust_threshold=0, progress=False)) == 1
    print([m for m in sys.modules if m.split(".")[0] == "brook" or m == "kimimaro.gpu"])
  """)
  assert out.strip() == "[]"

def test_gpu_module_imports_without_brook():
  run_python("""
    import sys
    sys.modules["brook"] = None # `import brook` now raises ImportError
    import kimimaro.gpu
  """)

def test_gpu_is_the_last_parameter_and_defaults_to_false():
  params = inspect.signature(kimimaro.skeletonize).parameters
  assert list(params)[-1] == "gpu"
  assert params["gpu"].default is False

# --- brook missing or wrong
def via_top_level(labels):
  return kimimaro.skeletonize(labels, gpu=True)

def via_gpu_module(labels):
  return kimimaro.gpu.skeletonize(labels, *[None] * 15) # import fails before the rest matters

@pytest.mark.parametrize("call", (via_top_level, via_gpu_module))
def test_missing_brook_raises_install_hint_chained_to_the_import_error(monkeypatch, labels, call):
  original = ImportError("no brook here")
  def find_spec(name, path=None, target=None):
    if name == "brook":
      raise original
  monkeypatch.delitem(sys.modules, "brook", raising=False)
  monkeypatch.setattr(
    sys, "meta_path", [types.SimpleNamespace(find_spec=find_spec)] + sys.meta_path
  )
  with pytest.raises(ImportError) as ei:
    call(labels)
  for part in INSTALL_HINT_PARTS:
    assert part in str(ei.value), part
  assert ei.value.__cause__ is original

@pytest.mark.parametrize("attrs", (
  {}, # an unrelated module that happens to be called brook
  {"skeletonize": lambda *args, **kwargs: {}}, # has skeletonize, but no brook.device
), ids=("no_skeletonize", "no_device_module"))
def test_wrong_brook_package_names_brook_cu12(monkeypatch, labels, attrs):
  impostor = types.ModuleType("brook")
  impostor.__path__ = []
  impostor.__dict__.update(attrs)
  monkeypatch.setitem(sys.modules, "brook", impostor)
  monkeypatch.setitem(sys.modules, "brook.device", None)
  with pytest.raises(ImportError, match="brook-cu12"):
    via_top_level(labels)

# --- require_gpu
def test_require_gpu_failure_propagates_and_skeletonize_is_never_called(fake_brook, labels):
  error = RuntimeError("no usable NVIDIA GPU")
  fake_brook.require_gpu_error = error
  with pytest.raises(RuntimeError) as ei:
    via_top_level(labels)
  assert ei.value is error
  assert fake_brook.events == ["require_gpu"]

def test_require_gpu_runs_before_skeletonize(fake_brook, labels):
  via_top_level(labels)
  assert fake_brook.events == ["require_gpu", "skeletonize"]

# --- argument forwarding
def test_forwards_exactly_the_documented_keywords_and_the_same_objects(fake_brook, labels):
  given = dict(
    teasar_params={"scale": 2.5, "const": 17}, anisotropy=(4.0, 4.0, 40.0),
    object_ids=[3, 5], dust_threshold=7,
    progress=False, fix_branching=False, in_place=True, fix_borders=False,
    extra_targets_before=[(1, 2, 3)], extra_targets_after=[(4, 5, 6)],
    fill_holes=True, fix_avocados=True,
    voxel_graph=np.ones((4, 4, 4), dtype=np.uint32),
  )
  result = kimimaro.skeletonize(
    labels, gpu=True, parallel=1, parallel_chunk_size=11, **given
  )
  assert result is fake_brook.result

  args, sent = fake_brook.only_call()
  assert len(args) == 1 and args[0] is labels
  assert sent.pop("streaming") is False # voxel_graph, fill_holes, fix_avocados
  assert set(sent) == BROOK_KEYWORDS # in particular no parallel, parallel_chunk_size
  for name in BROOK_KEYWORDS:
    assert sent[name] is given[name], name

@pytest.mark.parametrize("flipped", list(BOOL_DEFAULTS))
def test_each_boolean_option_is_forwarded_unflipped(fake_brook, labels, flipped):
  flags = dict(BOOL_DEFAULTS)
  flags[flipped] = not flags[flipped]
  kimimaro.skeletonize(labels, gpu=True, **flags)
  _, sent = fake_brook.only_call()
  for name, value in flags.items():
    assert sent[name] is value, name

@pytest.mark.parametrize("overrides,streaming_off", (
  ({"voxel_graph": np.ones((4, 4, 4), dtype=np.uint32)}, True), # array: `if voxel_graph` would raise
  ({"fill_holes": True}, True),
  ({"fix_avocados": True}, True),
  ({}, False),
), ids=("voxel_graph", "fill_holes", "fix_avocados", "none"))
def test_streaming_is_disabled_only_for_options_it_does_not_support(
  fake_brook, labels, overrides, streaming_off
):
  kimimaro.skeletonize(labels, gpu=True, **overrides)
  _, sent = fake_brook.only_call()
  if streaming_off:
    assert sent["streaming"] is False
  else:
    assert "streaming" not in sent

# --- parallel
@pytest.mark.parametrize("parallel", (0, 4)) # 0 means "all cores" in kimimaro
def test_parallel_other_than_one_warns_but_still_runs(fake_brook, labels, parallel):
  with pytest.warns(UserWarning, match=r"(?is)(?=.*parallel)(?=.*gpu)") as record:
    result = kimimaro.skeletonize(labels, gpu=True, parallel=parallel)
  assert result is fake_brook.result
  # attributed to the caller of kimimaro.skeletonize, not to kimimaro internals
  ours = [w for w in record if "parallel" in str(w.message)]
  assert len(ours) == 1
  assert os.path.abspath(ours[0].filename) == os.path.abspath(__file__)

def test_parallel_one_does_not_warn(fake_brook, labels):
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    kimimaro.skeletonize(labels, gpu=True, parallel=1)
  fake_brook.only_call()

# --- errors
def test_brook_dimension_error_becomes_kimimaro_dimension_error(fake_brook, labels):
  cause = fake_brook.DimensionError("Input must be 2D or 3D.")
  fake_brook.skeletonize_error = cause
  with pytest.raises(kimimaro.DimensionError) as ei:
    via_top_level(labels)
  assert ei.value.__cause__ is cause
  assert str(ei.value) == str(cause)

@pytest.mark.parametrize("error", (
  ValueError("bad value"), RuntimeError("out of memory"),
), ids=("value_error", "runtime_error"))
def test_other_brook_errors_propagate_unchanged(fake_brook, labels, error):
  fake_brook.skeletonize_error = error
  with pytest.raises(type(error)) as ei:
    via_top_level(labels)
  assert ei.value is error

# --- is_available
UNAVAILABLE = {
  "brook_missing": lambda monkeypatch, fake: block_brook(monkeypatch),
  "no_skeletonize": lambda monkeypatch, fake: delattr(fake.module, "skeletonize"),
  "no_devices": lambda monkeypatch, fake: setattr(fake, "device_count_value", 0),
  "device_count_raises": lambda monkeypatch, fake: setattr(
    fake, "device_count_error", RuntimeError("cuda init failed")),
  "require_gpu_raises": lambda monkeypatch, fake: setattr(
    fake, "require_gpu_error", RuntimeError("compute capability too old")),
}

@pytest.mark.parametrize("scenario", sorted(UNAVAILABLE))
def test_is_available_false_and_never_raises(monkeypatch, fake_brook, scenario):
  UNAVAILABLE[scenario](monkeypatch, fake_brook)
  assert kimimaro.gpu.is_available() is False

def test_is_available_true(fake_brook):
  assert kimimaro.gpu.is_available() is True
  assert fake_brook.calls == [] # probing must not run a skeletonization

# --- kimimaro.skeletonize(gpu=True) dispatch
def test_gpu_true_reaches_brook_with_kimimaro_defaults(fake_brook, labels):
  # (64 voxels <= dust_threshold: the CPU path would answer {} before any work)
  with warnings.catch_warnings():
    warnings.simplefilter("error") # the default parallel=1 must not warn
    result = kimimaro.skeletonize(labels, gpu=True)

  assert result is fake_brook.result
  args, sent = fake_brook.only_call()
  assert len(args) == 1 and args[0] is labels
  assert sent == dict(
    teasar_params=kimimaro.intake.DEFAULT_TEASAR_PARAMS, anisotropy=(1, 1, 1),
    object_ids=None, dust_threshold=1000,
    progress=True, fix_branching=True, in_place=False, fix_borders=True,
    extra_targets_before=[], extra_targets_after=[],
    fill_holes=False, fix_avocados=False, voxel_graph=None,
  )

def test_dispatch_happens_before_any_label_preprocessing(fake_brook):
  not_labels = object() # the CPU path would choke on it
  assert kimimaro.skeletonize(not_labels, gpu=True) is fake_brook.result
  assert fake_brook.only_call()[0][0] is not_labels

# --- packaging: pyproject.toml
@pytest.fixture(scope="module")
def pyproject():
  with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
    return tomllib.load(f)

@pytest.fixture
def gpu_requirement(pyproject):
  specs = pyproject["project"]["optional-dependencies"]["gpu"]
  assert len(specs) == 1
  return Requirement(specs[0])

def test_gpu_extra_requires_brook_cu12_0_1(gpu_requirement):
  assert canonicalize_name(gpu_requirement.name) == "brook-cu12"
  assert gpu_requirement.specifier.contains("0.1.0")
  assert not gpu_requirement.specifier.contains("0.2.0")

@pytest.mark.parametrize("platform,machine,python,expected", (
  ("linux", "x86_64", "3.12", True),
  ("linux", "x86_64", "3.13", True),
  ("linux", "x86_64", "3.11", False),
  ("linux", "aarch64", "3.12", False),
  ("darwin", "x86_64", "3.12", False), # only row that isolates the platform clause
  ("darwin", "arm64", "3.13", False),
  ("win32", "AMD64", "3.13", False),
))
def test_gpu_extra_marker_selects_only_linux_x86_64_python_3_12_plus(
  gpu_requirement, platform, machine, python, expected
):
  env = {"sys_platform": platform, "platform_machine": machine, "python_version": python}
  assert gpu_requirement.marker.evaluate(env) is expected

def test_brook_is_neither_a_core_dependency_nor_part_of_all(pyproject):
  project = pyproject["project"]
  for spec in project["dependencies"] + project["optional-dependencies"]["all"]:
    requirement = Requirement(spec)
    assert not requirement.name.lower().startswith("brook"), spec
    assert "gpu" not in requirement.extras, spec # no `kimimaro[gpu]` inside `all`

def test_gpu_pytest_marker_is_registered(pyproject):
  markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
  assert "gpu" in [m.split(":")[0].strip() for m in markers]

# --- Tier B: the real brook, no visible GPU. Never uses the stub above.
@pytest.fixture
def brook_without_gpu():
  real_brook()
  from brook.device import device_count # `import brook` does not load brook.device
  if device_count() != 0:
    pytest.skip("a CUDA device is visible; Tier B needs CUDA_VISIBLE_DEVICES=")

def test_no_gpu_skeletonize_raises_instead_of_falling_back(brook_without_gpu):
  # the CPU path would skeletonize this volume
  with pytest.raises(RuntimeError) as ei:
    kimimaro.skeletonize(tube(), gpu=True, dust_threshold=0, progress=False)
  assert str(ei.value).strip() != ""

def test_no_gpu_is_available_false(brook_without_gpu):
  assert kimimaro.gpu.is_available() is False

def test_no_gpu_cpu_path_still_works(brook_without_gpu):
  skels = kimimaro.skeletonize(tube(), dust_threshold=0, progress=False)
  assert list(skels) == [1]
  assert len(skels[1].vertices) > 1
