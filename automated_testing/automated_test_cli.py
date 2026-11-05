"""Tests for `kimimaro forge --gpu`; all but the last use a stub brook, no GPU."""
import os

import pytest

import numpy as np
from click.testing import CliRunner
from osteoid import Skeleton

import kimimaro
from kimimaro_cli import main

from automated_test_gpu import (
  INSTALL_HINT_PARTS, FakeBrook, block_brook, install_brook, run_python,
)
from automated_test_gpu_hw import gpu # fixture: skips without brook or a GPU

def forge(*args):
  return CliRunner().invoke(main, ["forge"] + list(args))

@pytest.fixture
def src(tmp_path):
  labels = np.zeros((48, 40, 36), dtype=np.uint32)
  labels[4:44, 10:14, 10:14] = 7      # a bar along x
  labels[20:24, 4:36, 24:28] = 42     # a bar along y
  labels[6:14, 24:34, 4:12] = 1001    # a block
  path = str(tmp_path / "labels.npy")
  np.save(path, labels)
  return path

def test_forge_help_lists_gpu():
  result = forge("--help")
  assert result.exit_code == 0
  assert "--gpu" in result.output

@pytest.mark.filterwarnings("ignore:.*parallel")
def test_forge_gpu_calls_brook_with_cli_arguments(monkeypatch, src, tmp_path):
  skel = Skeleton(
    np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
    np.array([[0, 1], [1, 2]], dtype=np.uint32), radii=np.ones(3, dtype=np.float32),
  )
  fake = FakeBrook()
  fake.result = {7: skel, 42: skel}
  install_brook(monkeypatch, fake)
  outdir = tmp_path / "out"

  result = forge(
    src, "--gpu", "-o", str(outdir), "--scale", "2.5", "--anisotropy", "4,4,40",
    "--dust", "7", "--fix-borders", "--no-fill-holes", "-p", "4",
  )
  assert result.exit_code == 0, result.output

  args, sent = fake.only_call()
  assert np.array_equal(args[0], np.load(src))
  expected = dict(
    anisotropy=(4, 4, 40), dust_threshold=7,
    fix_branching=True, fix_borders=True, fill_holes=False,
  )
  assert {key: sent[key] for key in expected} == expected
  assert sent["teasar_params"]["scale"] == 2.5
  assert "parallel" not in sent

  assert sorted(os.listdir(str(outdir))) == ["42.swc", "7.swc"]
  with open(str(outdir / "7.swc")) as f:
    assert len(Skeleton.from_swc(f.read()).vertices) == 3

# error None: brook is not installed; otherwise brook's require_gpu raises it
@pytest.mark.parametrize("error", (None, RuntimeError("no usable NVIDIA GPU")))
def test_forge_gpu_failure_is_a_clean_error(monkeypatch, src, tmp_path, error):
  if error is None:
    block_brook(monkeypatch)
    expected = INSTALL_HINT_PARTS
  else:
    fake = FakeBrook()
    fake.require_gpu_error = error
    install_brook(monkeypatch, fake)
    expected = (str(error),)

  result = forge(src, "--gpu", "-o", str(tmp_path / "out"))
  assert result.exit_code != 0
  assert isinstance(result.exception, SystemExit) # click error, not a crash
  for part in expected:
    assert part in result.output, part

def test_forge_cpu_errors_are_not_wrapped(monkeypatch, src):
  def boom(*args, **kwargs):
    raise RuntimeError("boom")
  monkeypatch.setattr(kimimaro, "skeletonize", boom)
  assert isinstance(forge(src).exception, RuntimeError)

@pytest.mark.parametrize("flags", ([], ["--no-gpu"]), ids=("default", "no_gpu"))
def test_forge_without_gpu_never_imports_brook(src, tmp_path, flags):
  out = run_python("""
    import sys
    from click.testing import CliRunner
    from kimimaro_cli import main
    args = ["forge", {src!r}, "-o", {outdir!r}, "--dust", "50"] + {flags!r}
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    print(sorted(
      name for name in sys.modules
      if name.split(".")[0] == "brook" or name == "kimimaro.gpu"
    ))
  """.format(src=src, outdir=str(tmp_path / "out"), flags=flags))
  assert out.strip().splitlines()[-1] == "[]"

@pytest.mark.gpu
def test_forge_gpu_writes_the_same_segids_as_cpu(gpu, src, tmp_path):
  names = {}
  for backend, flags in (("cpu", []), ("gpu", ["--gpu"])):
    outdir = str(tmp_path / backend)
    result = forge(src, "-o", outdir, "--dust", "50", "--anisotropy", "4,4,40", *flags)
    assert result.exit_code == 0, result.output
    names[backend] = set(os.listdir(outdir))
  assert names["gpu"] == names["cpu"] == {"7.swc", "42.swc", "1001.swc"}
