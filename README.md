[![PyPI version](https://badge.fury.io/py/kimimaro.svg)](https://badge.fury.io/py/kimimaro)  

# Kimimaro: Skeletonize Densely Labeled Images

Rapidly skeletonize all non-zero labels in 2D and 3D numpy arrays using a TEASAR derived method. The returned list of skeletons is in the format used by [cloud-volume](https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeletons). 

On a 3.7 GHz Intel i7 processor, this package processed a 512x512x100 volume with 333 labels in under a minute. It processed a 512x512x512 volume with 2124 labels in eight to thirteen minutes (depending on whether `fix_branching` is set).

<p style="font-style: italics;" align="center">
<img height=512 width=512 src="https://raw.githubusercontent.com/seung-lab/kimimaro/master/mass_skeletonization.png" alt="A Densely Labeled Volume Skeletonized with Kimimaro" /><br>
Fig. 1: A Densely Labeled Volume Skeletonized with Kimimaro
</p>

```python
import kimimaro

# A possible configuration for long thin axons
# without any somata in the field of view.
teasar_params = {
  # TEASAR parameters
  'scale': 4, # invalidation ball scale factor
  'const': 500, # invalidation ball const factor (in physical units)
  'pdrf_scale': 100000, # pdrf scale factor
  'pdrf_exponent': 4, # pdrf exponent

  # Special Soma handling, not applicable to this case
  'soma_detection_threshold': 99999999, # in physical units, set high to shut off
  'soma_acceptance_threshold': 99999999, # in physical units, set high to shut off
  'soma_invalidation_scale': 0.5, # Special invalidation ball for somata
  'soma_invalidation_const': 0, # Special invalidation ball for somata
}

labels = load_segmentation() # 3D labeled image array
skeletons = kimimaro.skeletonize(
  labels, teasar_params=teasar_params, 
  anisotropy=(32,32,40), # in nanometers
  dust_threshold=1000, # in voxels
  progress=True
)
```
*Detailed discussion of TEASAR parameters here or link to wiki.*

## `pip` Binary Installation

```bash
pip install kimimaro
```

## `pip` Manual Installation 

*Requires C++ compiler.*

```bash
sudo apt-get install python3-dev g++
pip3 install numpy
pip3 install kimimaro --no-binary :all:
```

## Motivation

The connectomics field commonly generates very large densely labeled volumes of neural tissue. Skeletons are one dimensional centerline representations of two or three dimensional objects. They have many uses, a few of which are for visualization of neurons, calculating global topological features, rapidly measuring electrical distances between objects, and imposing tree structures on neurons. There are several ways to compute skeletons and a few ways to define them. After some experimentation, we found that the TEASAR [1] approach gave fairly good results. Other approaches include topological thinning ("grass fire") and finding the centerline described by maximally inscribed spheres. Ignacio Arganda-Carreras, an alumnus of the Seung Lab, wrote a topological thinning plugin for Fiji called [Skeletonize3d](https://imagej.net/Skeletonize3D). 

There are several implementations of TEASAR used in the connectomics field, however it is commonly understood that implementations of TEASAR are slow and can use tens of gigabytes of memory. Our goal to skeletonize all labels in a petavoxel scale image quickly showed clear that existing sparse implementations are impractical. While adapting a sparse approach to a cloud pipeline, it was noticed that there were inefficiencies in CPU usage in the repeated evaluation of the Euclidean Distance Transform (EDT), the repeated evaluation of the connected components algorithm, in the construction of the graph used by Dijkstra's algorithm where the edges are implied by the spatial relationships between voxels, in the memory cost, quadratic in the number of voxels, of representing a graph that is implicit in image, and in the unnecessarily large data type used to represent relatively small cutouts. We also found that the naive implmentation of TEASAR's "rolling invalidation ball" unnecessarily reevaluated large numbers of voxels in a way that we could loosely characterize as quadratic in the skeleton path length.   

We found that commodity implementations of the EDT supported only binary images and did not support anisotropic dimensions (though many papers defining those techniques included anisotropic operation). Were unable to find any available Python or C++ libraries for performing Dijkstra's shortest path on an image. We also found that commodity implementations of connected components algorithms for images supported only binary images. Therefore, several libraries were devised to remedy these deficits. 

TBC

## TEASAR Algorithm and Deviations

TBC

### Using DAF for Targets, PDRF for Pathfinding

### Normalized DAF "Trickle Gradient"

### Rolling Invalidation Cube

Uses topological cues to perform O(V) invalidations instead of O(VN) where V is the number of voxels in the volume and N is the number of vertices in a skeleton path. This could be done as a sphere, it's just more time consuming to program.

### Soma Handling

We want to handle somas diff

## Related Projects

Several classic algorithms had to be specially tuned to make this module possible.  

1. [edt](https://github.com/seung-lab/euclidean-distance-transform-3d): A single pass, multi-label anisotropy supporting euclidean distance transform implementation. 
2. [dijkstra3d](https://github.com/seung-lab/dijkstra3d): Dijkstra's shortest-path algorithm defined on 26-connected 3D images. This avoids the time cost of edge generation and wasted memory of a graph representation.
3. [connected-components-3d](https://github.com/seung-lab/connected-components-3d): A connected components implementation defined on 26-connected 3D images with multiple labels.
4. [fastremap](https://github.com/seung-lab/fastremap): Allows high speed renumbering of labels from 1 in a 3D array in order to reduce memory consumption caused by unnecessarily large 32 and 64-bit labels.

This module was originally designed to be used with CloudVolume and Igneous. 

1. [CloudVolume](https://github.com/seung-lab/cloud-volume): Serverless client for reading and writing petascale chunked images of neural tissue, meshes, and skeletons.
2. [Igneous](https://github.com/seung-lab/igneous/tree/master/igneous): Distributed computation for visualizing connectomics datasets.  

Some of the TEASAR modifications used in this package were first demonstrated by Alex Bae.

1. [skeletonization](https://github.com/seung-lab/skeletonization): Python implementation of modified TEASAR for sparse labels.

## Credits

Alex Bae and William Silversmith

## References 

1. M. Sato, I. Bitter, M.A. Bender, A.E. Kaufman, and M. Nakajima. "TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust Skeletons". Proc. 8th Pacific Conf. on Computer Graphics and Applications. Oct. 2000. doi: 10.1109/PCCGA.2000.883951 ([link](https://ieeexplore.ieee.org/abstract/document/883951/))
2.  I. Bitter, A.E. Kaufman, and M. Sato. "Penalized-distance volumetric skeleton algorithm". IEEE Transactions on Visualization and Computer Graphics Vol. 7, Iss. 3, Jul-Sep 2001. doi: 10.1109/2945.942688 ([link](https://ieeexplore.ieee.org/abstract/document/942688/))
3. T. Zhao, S. Plaza. "Automatic Neuron Type Identification by Neurite Localization in the Drosophila Medulla". Sept. 2014. arXiv:1409.1892 [q-bio.NC] ([link](https://arxiv.org/abs/1409.1892))

