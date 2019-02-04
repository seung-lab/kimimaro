[![PyPI version](https://badge.fury.io/py/kimimaro.svg)](https://badge.fury.io/py/kimimaro)  

# Kimimaro: Skeletonize Densely Labeled Images

Rapidly skeletonize all non-zero labels in 2D and 3D numpy arrays using a TEASAR derived method. The returned list of skeletons is in the format used by [cloud-volume](https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeletons). 

On a 3.7 GHz Intel i7 processor, this package processed a 512x512x100 volume with 333 labels in under a minute. It processed a 512x512x512 volume with 2124 labels in eight to thirteen minutes (depending on whether `fix_branching` is set).

<p style="font-style: italics;" align="center">
<img height=512 width=512 src="https://raw.githubusercontent.com/seung-lab/kimimaro/master/mass_skeletonization.png" alt="A Densely Labeled Volume Skeletonized with Kimimaro" /><br>
Fig. 1: A Densely Labeled Volume Skeletonized with Kimimaro
</p>

## `pip` Installation 

*Requires C++ compiler.*

```bash
sudo apt-get install python3-dev g++
pip3 install numpy
pip3 install kimimaro 
```

In the future, we may create a fully binary distribution. 

## Example

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



## Motivation

The connectomics field commonly generates very large densely labeled volumes of neural tissue. Skeletons are one dimensional centerline representations of two or three dimensional objects. They have many uses, a few of which are for visualization of neurons, calculating global topological features, rapidly measuring electrical distances between objects, and imposing tree structures on neurons. There are several ways to compute skeletons and a few ways to define them. After some experimentation, we found that the TEASAR [1] approach gave fairly good results. Other approaches include topological thinning ("grass fire") and finding the centerline described by maximally inscribed spheres. Ignacio Arganda-Carreras, an alumnus of the Seung Lab, wrote a topological thinning plugin for Fiji called [Skeletonize3d](https://imagej.net/Skeletonize3D). 

There are several implementations of TEASAR used in the connectomics field, however it is commonly understood that implementations of TEASAR are slow and can use tens of gigabytes of memory. Our goal to skeletonize all labels in a petavoxel scale image quickly showed clear that existing sparse implementations are impractical. While adapting a sparse approach to a cloud pipeline, it was noticed that there were inefficiencies in CPU usage in the repeated evaluation of the Euclidean Distance Transform (EDT), the repeated evaluation of the connected components algorithm, in the construction of the graph used by Dijkstra's algorithm where the edges are implied by the spatial relationships between voxels, in the memory cost, quadratic in the number of voxels, of representing a graph that is implicit in image, and in the unnecessarily large data type used to represent relatively small cutouts. We also found that the naive implmentation of TEASAR's "rolling invalidation ball" unnecessarily reevaluated large numbers of voxels in a way that we could loosely characterize as quadratic in the skeleton path length.   

We found that commodity implementations of the EDT supported only binary images and did not support anisotropic dimensions (though many papers defining those techniques included anisotropic operation). Were unable to find any available Python or C++ libraries for performing Dijkstra's shortest path on an image. We also found that commodity implementations of connected components algorithms for images supported only binary images. Therefore, several libraries were devised to remedy these deficits. 

TBC

## Why TEASAR?

TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust skeletons, a 2000 paper by M. Sato and others [1], is a member of a family of algorithms that transform two and three dimensional structures into a one dimensional "skeleton" embedded in that higher dimension. One might concieve of a skeleton as extracting a stick figure drawing of a binary image. This problem is more difficult than it might seem. There are different ways one might concieve of such a drawing. For example, a stick drawing of a banana might merely be a curved centerline and a drawing of a doughnut might be a closed loop. In our case of analyzing neurons, sometimes we want the skeleton to include spines, short protrusions from dendrites that usually have synapses attached, and sometimes we want only the characterize the run length of the main trunk of a neurite.  

Additionally, data quality issues can be challenging as well. If one is skeletonizing a 2D image of a doughnut, but the angle were sufficiently declinated from the ring's orthogonal axis, would it even be possible to perform this task accurately? In a 3D case, if there are breaks or mergers in the labeling of a neuron, will the algorithm function sensibly? These issues are common in both manual and automatic image sementations.

In our problem domain of skeletonizing neurons from anisotropic voxel labels, our chosen algorithm should produce tree structures, handle fine or coarse detail extraction depending on the circumstances, handle voxel anisotropy, and be reasonably efficient in CPU and memory usage. TEASAR fufills these criteria. Notably, TEASAR doesn't guarantee the centeredness of the skeleton within the shape, but it makes an effort. The basic TEASAR algorithm is known to cut corners around turns and branch too early. A 2001 paper by members of the original TEASAR team describes a method for reducing the early branching issue on page 204, section 4.2.2. [2]

## TEASAR Derived Algorthm

We implemented TEASAR but made several important deviations from the published algorithm in order to improve path centeredness, increase performance, and handle bulging cell somas. We opted not to implement the gradient vector field step from [2] as our implementation is already quite fast. The paper claims a reduction of 70-85% in input voxels, so it might be worth investigating.  

In order to work with images that contain many labels, our general strategy is to perform as many actions as possible in such a way that all labels are treated in a single pass. Several of the component algorithms (e.g. connected components, euclidean distance transform) in our implementation can take several seconds to run per a pass, so it is important that they not be run hundreds or thousands of times. A large part of the engineering contribution of this package lies in the efficiency of these operations which reduce the runtime from the scale of hours to minutes.  

Given a 3D labeled voxel array, *I*, with N >= 0 labels, and ordered triple describing voxel anisotropy *A*, our algorithm can be divided into three phases, the pramble, skeletonization, and finalization in that order.

### Preamble

The Preamble takes a 3D image containing N labels and efficiently generates the connected components, distance transform, and bounding boxes needed by the skeletonization phase.

1. To enhance performance, if N is 0 return an empty set of skeletons.
2. Label the M connected components, *I<sub>cc</sub>*, of *I*.
3. To save memory, renumber the connected components in order from 1 to M. Adjust the data type of the new image to the smallest uint type that will contain M and overwrite *I<sub>cc</sub>*.
4. Generate a mapping of the renumbered *I<sub>cc</sub>* to *I* to assign meaningful labels to skeletons later on and delete *I* to save memory.
5. Compute *E*, the multi-label anisotropic Euclidean Distance Transform of *I<sub>cc</sub>* given *A*. *E* treats all interlabel edges as transform edges, but not the boundaries of the image. Black pixels are considered background.
6. Gather a list, *L<sub>cc</sub>* of unique labels from *I<sub>cc</sub>* and threshold which ones to process based on the number of voxels they represent to remove "dust".
7. In one pass, compute the list of bounding boxes, B, corresponding to each label in *L<sub>cc</sub>*.

### Skeletonization 

In this phase, we extract the tree structured skeleton from each connected component label.

### Finalization

In the final phase, we agglomerate the disparate connected component skeletons into single skeletons and assign their labels corresponding to the input image. This step is artificially broken out compared to how intermingled its implementation is with skeletonization, but it's conceptually separate.

### Using DAF for Targets, PDRF for Pathfinding

### Normalized DAF "Trickle Gradient"

### Rolling Invalidation Cube

Uses topological cues to perform O(V) invalidations instead of O(VN) where V is the number of voxels in the volume and N is the number of vertices in a skeleton path. This could be done as a sphere, it's just more time consuming to program.

### Soma Handling

We want to handle somas diff

### Zero Weighting Previous Paths

## Performance Tips

- If you only need a few labels skeletonized, pass in `object_ids` to bypass processing all the others. If `object_ids` contains only a single label, the masking operation will run faster.
- You may save on peak memory usage by using a `cc_safety_factor` < 1, only if you are sure the connected components algorithm will generate many fewer labels than there are pixels in your image.
- Larger TEASAR parameters scale and const require processing larger invalidation regions per path.
- Set `pdrf_exponent` to a small power of two (e.g. 1, 2, 4, 8, 16) for a small speedup.
- If you are willing to sacrifice the improved forking behavior, you can set `fix_branching=False` for a moderate 1.1x to 1.5x speedup (assuming your TEASAR parameters and data allow branching).

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
2. I. Bitter, A.E. Kaufman, and M. Sato. "Penalized-distance volumetric skeleton algorithm". IEEE Transactions on Visualization and Computer Graphics Vol. 7, Iss. 3, Jul-Sep 2001. doi: 10.1109/2945.942688 ([link](https://ieeexplore.ieee.org/abstract/document/942688/))
3. T. Zhao, S. Plaza. "Automatic Neuron Type Identification by Neurite Localization in the Drosophila Medulla". Sept. 2014. arXiv:1409.1892 \[q-bio.NC\] ([link](https://arxiv.org/abs/1409.1892))
4. A. Tagliasacchi, T. Delame, M. Spagnuolo, N. Amenta, A. Telea. "3D Skeletons: A State-of-the-Art Report". May 2016. Computer Graphics Forum. Vol. 35, Iss. 2. https://doi.org/10.1111/cgf.12865
