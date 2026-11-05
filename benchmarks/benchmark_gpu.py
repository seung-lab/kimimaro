"""
Time kimimaro.skeletonize on the CPU or on the GPU (gpu=True), or compare two saved results:
  python benchmarks/benchmark_gpu.py VOLUME [--crop N] --backend cpu|gpu [--parallel P]
         [--runs 3] [--no-fix-branching] [--save out.npz]
  python benchmarks/benchmark_gpu.py --compare a.npz b.npz

VOLUME is a .npy or connectomics.npy.ckl.gz (crackle), converted to Fortran order. The parameters are
those of benchmark.py. Each timing is one complete call (NumPy labels in, skeletons out), measured with
time.perf_counter. The first call is the cold start (CUDA initialization on the GPU) and is reported
separately; the --runs calls after it are summarized by median, min and max (--runs 0: cold start only).
--backend gpu needs: pip install "kimimaro[gpu]"
"""
import argparse
import statistics
import sys
import time

import crackle
import numpy as np
from scipy.spatial import cKDTree

TEASAR_PARAMS = dict( # const and soma_* are in physical units
  scale=1.5, const=300, pdrf_exponent=4, pdrf_scale=100000,
  soma_detection_threshold=1100, soma_acceptance_threshold=3500,
  soma_invalidation_scale=1.0, soma_invalidation_const=300,
)
ANISOTROPY = (16, 16, 40)

def load(path, crop):
  labels = np.load(path) if path.endswith(".npy") else crackle.load(path)
  return np.asfortranarray(labels[:crop, :crop, :crop] if crop else labels)

def cable(vertices, edges):
  """Cable length of one skeleton, in physical units."""
  v = vertices.astype(np.float64)
  return np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1).sum()

def benchmark(args):
  if args.backend == "gpu":
    try:
      import brook.device # imported here so that the cold start is CUDA initialization only
    except ImportError as err:
      sys.exit(f'error: --backend gpu needs the Brook GPU backend ({err}): pip install "kimimaro[gpu]"')
  import kimimaro

  labels = load(args.volume, args.crop)
  options = dict(
    teasar_params=TEASAR_PARAMS, anisotropy=ANISOTROPY, dust_threshold=1000,
    fix_borders=True, fix_branching=args.fix_branching, progress=False,
    gpu=(args.backend == "gpu"), parallel=args.parallel, # parallel is ignored when gpu=True
  )
  print(f"{args.backend}, volume {labels.shape} {labels.dtype}, fix_branching={args.fix_branching}")

  times = []
  for i in range(args.runs + 1):
    start = time.perf_counter()
    skels = kimimaro.skeletonize(labels, **options)
    times.append(time.perf_counter() - start)
    print(f"{'cold start' if i == 0 else 'run %d' % i}: {times[-1]:.3f} s")
  runs = times[1:]
  if runs:
    print(f"median {statistics.median(runs):.3f} s, min {min(runs):.3f} s, max {max(runs):.3f} s")

  skels = { label: (np.asarray(s.vertices), np.asarray(s.edges)) for label, s in skels.items() }
  print(f"output: {len(skels)} labels, {sum(len(v) for v, _ in skels.values())} vertices, "
        f"cable length {sum(cable(*s) for s in skels.values()):.6g} (physical units)")
  if args.save: # per label, v<label> and e<label>: vertices (physical units) and edges
    np.savez_compressed(args.save, **{ f"{k}{label}": a for label, s in skels.items() for k, a in zip("ve", s) })

def read(path):
  with np.load(path) as z:
    return { int(k[1:]): (z[k], z["e" + k[1:]]) for k in z.files if k[0] == "v" }

def compare(path_a, path_b):
  A, B = read(path_a), read(path_b)
  cable_a = { label: cable(*s) for label, s in A.items() }
  cable_b = { label: cable(*s) for label, s in B.items() }
  common = sorted(A.keys() & B.keys())
  if not common:
    sys.exit("error: the files have no label in common")

  delta, percent, distances = {}, [], []
  for label in common:
    delta[label] = abs(cable_b[label] - cable_a[label])
    if cable_a[label] > 0:
      percent.append(100 * delta[label] / cable_a[label])
    pa, pb = A[label][0] / np.array(ANISOTROPY), B[label][0] / np.array(ANISOTROPY) # in voxels
    distances += [ cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0] ] # nearest vertex, both ways
  distances = np.concatenate(distances)
  total_a, total_b, worst = sum(cable_a.values()), sum(cable_b.values()), max(delta, key=delta.get)

  print(f"same label set: {A.keys() == B.keys()} ({len(common)} common, "
        f"{len(A.keys() - B.keys())} only in A, {len(B.keys() - A.keys())} only in B)")
  print(f"total cable length change, B vs A: {100 * (total_b - total_a) / total_a:+.2f}%")
  print(f"per-object |cable delta|: median {np.median(percent):.2f}%, p95 {np.percentile(percent, 95):.2f}%")
  print(f"nearest-vertex distance, voxels (both ways, pooled): "
        f"mean {distances.mean():.3f}, p95 {np.percentile(distances, 95):.3f}")
  print(f"largest cable change: label {worst}, {100 * (cable_b[worst] - cable_a[worst]) / cable_a[worst]:+.2f}%")

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("volume", nargs="?", help=".npy or .ckl.gz label volume")
  parser.add_argument("--crop", type=int, metavar="N", help="use volume[:N, :N, :N]")
  parser.add_argument("--backend", choices=("cpu", "gpu"))
  parser.add_argument("--parallel", type=int, default=1, metavar="P", help="cpu processes (default 1; <= 0: all cores)")
  parser.add_argument("--runs", type=int, default=3, help="timed runs after the cold start (default 3; 0: only the cold start)")
  parser.add_argument("--no-fix-branching", dest="fix_branching", action="store_false")
  parser.add_argument("--save", metavar="OUT.npz", help="save the last result for --compare")
  parser.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"), help="compare B against A")
  args = parser.parse_args()

  if args.compare:
    compare(*args.compare)
  elif not (args.volume and args.backend) or args.runs < 0:
    parser.error("VOLUME, --backend and --runs >= 0 are required (or --compare A.npz B.npz)")
  else:
    benchmark(args)

if __name__ == "__main__":
  main()
