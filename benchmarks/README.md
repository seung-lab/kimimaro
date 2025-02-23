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


