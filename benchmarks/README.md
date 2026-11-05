Benchmarks
==========

To open `connectomics.npy.ckl.gz` you must use [`crackle-codec`](https://github.com/seung-lab/crackle).

Except where noted, these benchmarks were executed on an 2.8 GHz Dual-Core Intel Core i7 with 1600 MHz DDR3 RAM. The data source used was `connectomics.npy` which can be found in this repository. `connectomics.npy` is a 32-bit 512x512x512 cutout of mouse visual cortex at 16nm x 16nm x 40nm resolution that contains 2124 connected components including a partial cell body and a large glia fragment.

Below, we compared the run time and peak memory usage of Kimimaro across many versions that contained performance significant updates. Due to the annoying length of each run, each value represents a single run, so there is some random perturbation around the true mean that can obscure the value of small improvements. Version 0.4.2 can be considered the first "feature complete" version that includes quality improvements like fix_branches, fix_borders, and a reasonable root selected for the cell body.

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/kimimaro/master/benchmarks/kimimaro-execution-time-by-version.png" alt="Kimimaro Execution Time by Version on connectomics.npy" /><br>
Fig. 1: Kimimaro Execution Time by Version on `connectomics.npy`
</p>

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/kimimaro/master/benchmarks/kimimaro-peak-memory-usage-by-version.png" alt="Kimimaro Peak Memory Usage by Version on connectomics.npy" /><br>
Fig. 2: Kimimaro Peak Memory Usage by Version on `connectomics.npy`
</p>

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/kimimaro/master/benchmarks/kimimaro-memory-profiles-0.1.0-3.0.0.png" alt="Kimimaro Memory Profile Versions 0.3.1 vs. 3.0.0" /><br>
Fig. 3: Kimimaro Memory Profile Versions (blue) 0.3.1 (black) 3.0.0. The first hump on the left is processing a soma. The second hump is a glia.
</p>

GPU Backend
-----------

`benchmark_gpu.py` times `kimimaro.skeletonize` on the CPU against `kimimaro.skeletonize(..., gpu=True)`, which runs on an NVIDIA GPU through [Brook](https://github.com/giorgioangel/brook) (`pip install "kimimaro[gpu]"`). It uses the `teasar_params` of `benchmark.py` with `anisotropy=(16,16,40)`. The volume is a `.npy` file or `connectomics.npy.ckl.gz`; `--crop N` takes `[:N,:N,:N]`.

```bash
V=benchmarks/connectomics.npy.ckl.gz
python benchmarks/benchmark_gpu.py $V --backend gpu --save gpu.npz
python benchmarks/benchmark_gpu.py $V --backend cpu --parallel 32 --runs 0 --save cpu32.npz
taskset -c 0 python benchmarks/benchmark_gpu.py $V --backend cpu --parallel 1 --runs 0
python benchmarks/benchmark_gpu.py --compare cpu32.npz gpu.npz
# --crop 256 for the 256^3 crop, --no-fix-branching to turn branching correction off
```

Each timing is one complete call, NumPy labels in and skeletons out. The first call in the process is the cold start (lazy imports and, on the GPU, CUDA initialization) and is reported separately; `--runs` more calls follow (3 by default, `--runs 0` for none) and are summarized by median, min and max. `--compare` reports whether the label sets match, the change in total cable length, the median and 95th percentile of the per-object cable length change, the mean and 95th percentile of the nearest-vertex distance between the skeletons of each object (in voxels), and the label whose cable length changed most.

### GPU Results

Measured with `benchmark_gpu.py` on 2026-10-02 with Kimimaro 5.8.5 and Brook 0.1.0 on Python 3.12, an Intel Core i9-14900KF (24 cores, 32 threads) and an NVIDIA GeForce RTX 4090 with driver 615.71. The machine was a desktop workstation with a graphical session running, so treat the timings as indicative. The input is the 512x512x512 volume above or its `[:N,:N,:N]` crops, with the `teasar_params` of `benchmark.py` and `anisotropy=(16,16,40)`. The GPU columns are the median of three runs after the cold start; the CPU columns are a single call (`--runs 0`), with `taskset -c 0` for one core and `--parallel 32` for 32 workers.

| Volume | `fix_branching` | CPU, 1 core | CPU, 32 workers | GPU | GPU, cold start | Speedup vs 1 core | vs 32 workers |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256^3 | on | 28.4 s | 25.5 s | 0.38 s | 0.60 s | 74x | 66x |
| 256^3 | off | 26.0 s | 22.9 s | 0.38 s | 0.55 s | 69x | 61x |
| 384^3 | on | 152.6 s | 94.1 s | 1.62 s | 1.61 s | 94x | 58x |
| 384^3 | off | 88.0 s | 52.0 s | 1.15 s | 1.34 s | 77x | 45x |
| 512^3 | on | 855.6 s | 409.5 s | 2.98 s | 3.17 s | 287x | 137x |
| 512^3 | off | 236.5 s | 107.3 s | 2.59 s | 2.79 s | 91x | 41x |

- The cold start includes CUDA initialization and loading the GPU code, so it is slower than the steady state for small volumes.
- The CPU runs with 1 and with 32 workers returned identical skeletons.
- The GPU used about 5.6 GiB of GPU memory at 512^3 with `fix_branching` on and 5.2 GiB with it off (as reported by `nvidia-smi` during the run).
- Speedups depend on the shapes in the volume, the parameters and the hardware.

The GPU skeletons are close to, but not identical to, the CPU ones (see "Differences from CPU mode" in the main README). All labels were found in every run, and per object the cable length differed by a median of 0.05% to 0.96% (95th percentile 2.9% to 4.4%), with a mean nearest-vertex distance of 0.06 to 0.10 voxel between the two skeletons of an object. With `fix_branching` on, the total cable length of the larger volumes differs much more (-14.0% at 512^3, -27.5% at 384^3, against -0.65% at 256^3) because of one soma, label 25024949: its GPU skeleton is 85% shorter than the CPU one (about a seventh of the cable length) with almost the same number of vertices, because Kimimaro 5.8.5 adds long straight edges from the soma root to many branches. With `fix_branching` off the same volumes differ by -1.3% (512^3) and -1.2% (384^3), and that soma by +0.3% at 512^3.
