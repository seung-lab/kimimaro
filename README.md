[![Build Status](https://travis-ci.org/seung-lab/kimimaro.svg?branch=master)](https://travis-ci.org/seung-lab/kimimaro) [![PyPI version](https://badge.fury.io/py/kimimaro.svg)](https://badge.fury.io/py/kimimaro)  

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

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/kimimaro/master/kimimaro_512x512x512_benchmark.png" alt="A Densely Labeled Volume Skeletonized with Kimimaro" /><br>
Fig. 2: Memory Usage on a 512x512x512 Densely Labeled Volume
</p>

Figure 2 shows the memory usage and processessing time (~390 seconds, about 6.5 minutes) required when Kimimaro 0.5.2 was applied to a 512x512x512 cutout, *labels*, from a connectomics dataset containing 2124 connected components. The different sections of the algorithm are depicted. Grossly, the preamble runs for about half a minute, skeletonization for about six minutes, and finalization within seconds. The peak memory usage was about 4.1 GB. The code below was used to process *labels*. The processing of the glia was truncated in due to a combination of *fix_borders* and max_paths.  

Kimimaro has come a long way. Version 0.2.1 took over 15 minutes and had a  Preamble run time twice as long on the same dataset.    

```python
# LISTING 1: Producing Skeletons from a labeled image.

import kimimaro

labels = np.load(...)

skels = kimimaro.skeletonize(
  labels, 
  teasar_params={
    'scale': 4,
    'const': 500, # physical units
    'pdrf_exponent': 4,
    'pdrf_scale': 100000,
    'soma_detection_threshold': 1100, # physical units
    'soma_acceptance_threshold': 3500, # physical units
    'soma_invalidation_scale': 1.0,
    'soma_invalidation_const': 300, # physical units
    'max_paths': 50, # default None
  },
  # object_ids=[ ... ], # process only the specified labels
  dust_threshold=1000, # skip connected components with fewer than this many voxels
  anisotropy=(16,16,40), # default True
  fix_branching=True, # default True
  fix_borders=True, # default True
  progress=True, # default False, show progress bar
  parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
)

# LISTING 2: Combining skeletons produced from 
#            adjacent or overlapping images.

import kimimaro
from cloudvolume import PrecomputedSkeleton

skels = ... # a set of skeletons produced from the same label id
skel = PrecomputedSkeleton.simple_merge(skels).consolidate()
skel = kimimaro.postprocess(
  skel, 
  dust_threshold=1000, # physical units
  tick_threshold=3500 # physical units
)

# Split input skeletons into connected components and
# then join the two nearest vertices within `radius` distance
# of each other until there is only a single connected component
# or no pairs of points nearer than `radius` exist. 
# Fuse all remaining components into a single skeleton.
skel = kimimaro.join_close_components([skel1, skel2], radius=1500) # 1500 units threshold
skel = kimimaro.join_close_components([skel1, skel2], radius=None) # no threshold
```

### Tweaking `kimimaro.skeletonize` Parameters

This algorithm works by finding a root point on a 3D object and then serially tracing paths via dijksta's shortest path algorithm through a penalty field to the most distant unvisited point. After each pass, there is a sphere (really a circumscribing cube) that expands around each vertex in the current path that marks part of the object as visited.  

For a visual tutorial on the basics of the skeletonization procedure, check out this wiki article: [A Pictorial Guide to TEASAR Skeletonization](https://github.com/seung-lab/kimimaro/wiki/A-Pictorial-Guide-to-TEASAR-Skeletonization)

For more detailed information, [read below](https://github.com/seung-lab/kimimaro#ii-skeletonization) or the [TEASAR paper](https://ieeexplore.ieee.org/abstract/document/883951/) (though we [deviate from TEASAR](https://github.com/seung-lab/kimimaro#teasar-derived-algorthm) in a few places). [1]

#### `scale` and `const`

Usually, the most important parameters to tweak are `scale` and `const` which control the radius of this invalidation sphere according to the equation `r(x,y,z) = scale * DBF(x,y,z) + const` where the dimensions are physical (e.g. nanometers, i.e. corrected for anisotropy). `DBF(x,y,z)` is the physical distance from the shape boundary at that point.  

Check out this [wiki article](https://github.com/seung-lab/kimimaro/wiki/Intuition-for-Setting-Parameters-const-and-scale) to help refine your intuition.

#### `anisotropy`

Represents the physical dimension of each voxel. For example, a connectomics dataset might be scanned with an electron microscope at 4nm x 4nm per pixel and stacked in slices 40nm thick. i.e. `anisotropy=(4,4,40)`. You can use any units so long as you are consistent.

#### `dust_threshold`

This threshold culls connected components that are smaller than this many voxels.  

#### `max_paths`  

Limits the number of paths that can be drawn for the given label. Certain cells, such as glia, that may not be important for the current analysis may be expensive to process and can be aborted early.  

#### `pdrf_scale` and `pdrf_exponent`

The `pdrf_scale` and `pdrf_exponent` represent parameters to the penalty equation that takes the euclidean distance field (**D**) and augments it so that cutting closer to the border is very penalized to make dijkstra take paths that are more centered.   

P<sub>r</sub> = `pdrf_scale` * (1 - **D** / max(**D**)) <sup>`pdrf_exponent`</sup> + (directional gradient < 1.0).  

The default settings should work fairly well, but under large anisotropies or with cavernous morphologies, it's possible that you might need to tweak it. If you see the skeleton go haywire inside a large area, it could be a collapse of floating point precision.  

#### `soma_acceptance_threshold` and `soma_detection_threshold`

We process somas specially because they do not have a tubular geometry and instead should be represented in a hub and spoke manner. `soma_acceptance_threshold` is the physical radius (e.g. in nanometers) beyond which we classify a connected component of the image as containing a soma. The distance transform's output is depressed by holes in the label, which are frequently produced by segmentation algorithms on somata. We can fill them, but the hole filling algorithm we use is slow so we would like to only apply it occasionally. Therefore, we set a lower threshold, the `soma_acceptance_threshold`, beyond which we fill the holes and retest the soma.  

#### `soma_invalidation_scale` and `soma_invalidation_const`   

Once we have classified a region as a soma, we fix root of the skeletonization algorithm at one of the  points of maximum distance from the boundary (usually there is only one). We then mark as visited all voxels around that point in a spherical radius described by `r(x,y,z) = soma_invalidation_scale * DBF(x,y,z) + soma_invalidation_const` where DBF(x,y,z) is the physical distance from the shape boundary at that point. If done correctly, this can prevent skeletons from being drawn to the boundaries of the soma, and instead pulls the skeletons mainly into the processes extending from the cell body.  

#### `fix_borders`

This feature makes it easier to connect the skeletons of adjacent image volumes that do not fit in RAM. If enabled, skeletons will be deterministically drawn to the approximate center of the 2D contact area of each place where the shape contacts the border. This can affect the performance of the operation positively or negatively depending on the shape and number of contacts.  

#### `fix_branching`  

You'll probably never want to disable this, but base TEASAR is infamous for forking the skeleton at branch points way too early. This option makes it preferential to fork at a more reasonable place at a significant performance penalty. 

#### `progress`

Show a progress bar once the skeletonization phase begins.

#### `parallel`  

Use a pool of processors to skeletonize faster. Each process allocatable task is the skeletonization of one connected component (so it won't help with a single label that takes a long time to skeletonize). This option also affects the speed of the initial euclidean distance transform, which is parallel enabled and is the most expensive part of the Preamble (described below).  

### Performance Tips

- If you only need a few labels skeletonized, pass in `object_ids` to bypass processing all the others. If `object_ids` contains only a single label, the masking operation will run faster.
- You may save on peak memory usage by using a `cc_safety_factor` < 1, only if you are sure the connected components algorithm will generate many fewer labels than there are pixels in your image.
- Larger TEASAR parameters scale and const require processing larger invalidation regions per path.
- Set `pdrf_exponent` to a small power of two (e.g. 1, 2, 4, 8, 16) for a small speedup.
- If you are willing to sacrifice the improved branching behavior, you can set `fix_branching=False` for a moderate 1.1x to 1.5x speedup (assuming your TEASAR parameters and data allow branching).
- If your dataset contains important cells (that may in fact be the seat of consciousness) but they take significant processing power to analyze, you can save them to savor for later by setting `max_paths` to some reasonable level which will abort and proceed to the next label after the algorithm detects that that at least that many paths will be needed.
- Parallel distributes work across connected components and is generally a good idea if you have the cores and memory. Not only does it make single runs proceed faster, but you can also practically use a much larger context; that improves soma processing as they are less likely to be cut off. The Preamble of the algorithm (detailed below) is still single threaded at the moment, so task latency increases with size. 

## Motivation

The connectomics field commonly generates very large densely labeled volumes of neural tissue. Skeletons are one dimensional representations of two or three dimensional objects. They have many uses, a few of which are visualization of neurons, calculating global topological features, rapidly measuring electrical distances between objects, and imposing tree structures on neurons (useful for computation and user interfaces). There are several ways to compute skeletons and a few ways to define them [4]. After some experimentation, we found that the TEASAR [1] approach gave fairly good results. Other approaches include topological thinning ("onion peeling") and finding the centerline described by maximally inscribed spheres. Ignacio Arganda-Carreras, an alumnus of the Seung Lab, wrote a topological thinning plugin for Fiji called [Skeletonize3d](https://imagej.net/Skeletonize3D). 

There are several implementations of TEASAR used in the connectomics field [3][5], however it is commonly understood that implementations of TEASAR are slow and can use tens of gigabytes of memory. Our goal to skeletonize all labels in a petavoxel scale image quickly showed clear that existing sparse implementations are impractical. While adapting a sparse approach to a cloud pipeline, we noticed that there are inefficiencies in repeated evaluation of the Euclidean Distance Transform (EDT), the repeated evaluation of the connected components algorithm, in the construction of the graph used by Dijkstra's algorithm where the edges are implied by the spatial relationships between voxels, in the memory cost, quadratic in the number of voxels, of representing a graph that is implicit in image, in the unnecessarily large data type used to represent relatively small cutouts, and in the repeated downloading of overlapping regions. We also found that the naive implmentation of TEASAR's "rolling invalidation ball" unnecessarily reevaluated large numbers of voxels in a way that could be loosely characterized as quadratic in the skeleton path length.   

We further found that commodity implementations of the EDT supported only binary images. We were unable to find any available Python or C++ libraries for performing Dijkstra's shortest path on an image. Commodity implementations of connected components algorithms for images supported only binary images. Therefore, several libraries were devised to remedy these deficits (see Related Projects). 

## Why TEASAR?

TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust skeletons, a 2000 paper by M. Sato and others [1], is a member of a family of algorithms that transform two and three dimensional structures into a one dimensional "skeleton" embedded in that higher dimension. One might concieve of a skeleton as extracting a stick figure drawing from a binary image. This problem is more difficult than it might seem. There are different situations one must consider when making such a drawing. For example, a stick drawing of a banana might merely be a curved centerline and a drawing of a doughnut might be a closed loop. In our case of analyzing neurons, sometimes we want the skeleton to include spines, short protrusions from dendrites that usually have synapses attached, and sometimes we want only the characterize the run length of the main trunk of a neurite.  

Additionally, data quality issues can be challenging as well. If one is skeletonizing a 2D image of a doughnut, but the angle were sufficiently declinated from the ring's orthogonal axis, would it even be possible to perform this task accurately? In a 3D case, if there are breaks or mergers in the labeling of a neuron, will the algorithm function sensibly? These issues are common in both manual and automatic image sementations.

In our problem domain of skeletonizing neurons from anisotropic voxel labels, our chosen algorithm should produce tree structures, handle fine or coarse detail extraction depending on the circumstances, handle voxel anisotropy, and be reasonably efficient in CPU and memory usage. TEASAR fufills these criteria. Notably, TEASAR doesn't guarantee the centeredness of the skeleton within the shape, but it makes an effort. The basic TEASAR algorithm is known to cut corners around turns and branch too early. A 2001 paper by members of the original TEASAR team describes a method for reducing the early branching issue on page 204, section 4.2.2. [2]

## TEASAR Derived Algorthm

We implemented TEASAR but made several deviations from the published algorithm in order to improve path centeredness, increase performance, handle bulging cell somas, and enable efficient chunked evaluation of large images. We opted not to implement the gradient vector field step from [2] as our implementation is already quite fast. The paper claims a reduction of 70-85% in input voxels, so it might be worth investigating.  

In order to work with images that contain many labels, our general strategy is to perform as many actions as possible in such a way that all labels are treated in a single pass. Several of the component algorithms (e.g. connected components, euclidean distance transform) in our implementation can take several seconds per a pass, so it is important that they not be run hundreds or thousands of times. A large part of the engineering contribution of this package lies in the efficiency of these operations which reduce the runtime from the scale of hours to minutes.  

Given a 3D labeled voxel array, *I*, with N >= 0 labels, and ordered triple describing voxel anisotropy *A*, our algorithm can be divided into three phases, the pramble, skeletonization, and finalization in that order.

### I. Preamble

The Preamble takes a 3D image containing *N* labels and efficiently generates the connected components, distance transform, and bounding boxes needed by the skeletonization phase.

1. To enhance performance, if *N* is 0 return an empty set of skeletons.
2. Label the M connected components, *I<sub>cc</sub>*, of *I*.
3. To save memory, renumber the connected components in order from 1 to *M*. Adjust the data type of the new image to the smallest uint type that will contain *M* and overwrite *I<sub>cc</sub>*.
4. Generate a mapping of the renumbered *I<sub>cc</sub>* to *I* to assign meaningful labels to skeletons later on and delete *I* to save memory.
5. Compute *E*, the multi-label anisotropic Euclidean Distance Transform of *I<sub>cc</sub>* given *A*. *E* treats all interlabel edges as transform edges, but not the boundaries of the image. Black pixels are considered background.
6. Gather a list, *L<sub>cc</sub>* of unique labels from *I<sub>cc</sub>* and threshold which ones to process based on the number of voxels they represent to remove "dust".
7. In one pass, compute the list of bounding boxes, *B*, corresponding to each label in *L<sub>cc</sub>*.

### II. Skeletonization 

In this phase, we extract the tree structured skeleton from each connected component label. Below, we reference variables defined in the Preamble. For clarity, we omit the soma specific processing and hold `fix_branching=True`. 

For each label *l* in *L<sub>cc</sub>* and *B*...

1. Extract *I<sub>l</sub>*, the cropped binary image tightly enclosing *l* from *I<sub>cc</sub>* using *B<sub>l</sub>*
2. Using *I<sub>l</sub>* and *B<sub>l</sub>*, extract *E<sub>l</sub>* from *E*. *E<sub>l</sub>* is the cropped tightly enclosed EDT of *l*. This is much faster than recomputing the EDT for each binary image.
3. Find an arbitrary foreground voxel and using that point as a source, compute the anisotropic euclidean distance field for *I<sub>l</sub>*. The coordinate of the maximum value is now "the root" *r*.
4. From *r*, compute the euclidean distance field and save it as the distance from root field *D<sub>r</sub>*.
5. Compute the penalized distance from root field *P<sub>r</sub>* = `pdrf_scale` * ((1 - *E<sub>l</sub>* / max(*E<sub>l</sub>*)) ^ `pdrf_exponent`) + *D<sub>r</sub>* / max(*D<sub>r</sub>*). 
6. While *I<sub>l</sub>* contains foreground voxels:
    1. Identify a target coordinate, *t*, as the foreground voxel with maximum distance in *D<sub>r</sub>* from *r*.
    2. Draw the shortest path *p* from *r* to *t* considering the voxel values in *P<sub>r</sub>* as edge weights.
    3. For each vertex *v* in *p*, extend an invalidation cube of physical side length computed as `scale` * *E<sub>l</sub>*(*v*) + `const` and convert any foreground pixels in *I<sub>l</sub>* that overlap with these cubes to background pixels.
    4. (Only if `fix_branching=True`) For each vertex coordinate *v* in *p*, set *P<sub>r</sub>*(*v*) = 0.
    5. Append *p* to a list of paths for this label.
7. Using *E<sub>l</sub>*, extract the distance to the nearest boundary each vertex in the skeleton represents.
8. For each raw skeleton extracted from *I<sub>l</sub>*, translate the vertices by *B<sub>l</sub>* to correct for the translation the cropping operation induced.
9. Multiply the vertices by the anisotropy *A* to place them in physical space.

If soma processing is considered, we modify the root (*r*) search process as follows:  

1. If max(*E<sub>l</sub>*) > `soma_detection_threshold`...
  1. Fill toplogical holes in *I<sub>l</sub>*. Soma are large regions that often have dust from imperfect automatic labeling methods.
  2. Recompute *E<sub>l</sub>* from this cleaned up image.
  3. If max(*E<sub>l</sub>*) > `soma_acceptance_threshold`, divert to soma processing mode.
2. If in soma processing mode, continue, else go to step 3 in the algorithm above.
3. Set *r* to the coordinate corresponding to max(*E<sub>l</sub>*)
4. Create an invalidation sphere of physical radius `soma_invalidation_scale` * max(*E<sub>l</sub>*) + `soma_invalidation_const` and erase foreground voxels from *I<sub>l</sub>* contained within it. This helps prevent errant paths from being drawn all over the soma.
5. Continue from step 4 in the above algorithm.

### III. Finalization

In the final phase, we agglomerate the disparate connected component skeletons into single skeletons and assign their labels corresponding to the input image. This step is artificially broken out compared to how intermingled its implementation is with skeletonization, but it's conceptually separate.

## Deviations from TEASAR

There were several places where we took a different approach than called for by the TEASAR authors.

### Using DAF for Targets, PDRF for Pathfinding

The original TEASAR algorithm defines the Penalized Distance from Root voxel Field (PDRF, *P<sub>r</sub>* above) as:

```
PDRF = 5000 * (1 - DBF / max(DBF))^16 + DAF
```

DBF is the Distance from Boundary Field (*E<sub>l</sub>* above) and DAF is the Distance from Any voxel Field (*D<sub>r</sub>* above).  

We found the addition of the DAF tended to perturb the skeleton path from the centerline better described by the inverted DBF alone. We also found it helpful to modify the constant and exponent to tune cornering behavior. Initially, we completely stripped out the addition of the DAF from the PDRF, but this introduced a different kind of problem. The exponentiation of the PDRF caused floating point values to collapse in wide open spaces. This made the skeletons go crazy as they traced out a path described by floating point errors.  

The DAF provides a very helpful gradient to follow between the root and the target voxel, we just don't want that gradient to knock the path off the centerline. Therefore, in light of the fact that the PDRF base field is very large, we add the normalized DAF which is just enough to overwhelm floating point errors and provide direction in wide tubes and bulges.  

The original paper also called for selecting targets using the max(PDRF) foreground values. However, this is a bit strange since the PDRF values are dominated by boundary effects rather than a pure distance metric. Therefore, we select targets from the max(DAF) forground value.

### Zero Weighting Previous Paths (`fix_branching=True`)

The 2001 skeletonization paper [2] called for correcting early forking by computing a DAF using already computed path vertices as field sources. This allows Dijkstra's algorithm to trace the existing path cost free and diverge from it at a closer point to the target.  

As we have strongly deemphasized the role of the DAF in dijkstra path finding, computing this field is unnecessary and we only need to set the PDRF to zero along the path of existing skeletons to achieve this effect. This saves us an expensive repeated DAF calculation per path.  

However, we still incur a substantial cost for taking this approach because we had been computing a dijkstra "parental field" that recorded the shortest path to the root from every foreground voxel. We then used this saved result to rapidly compute all paths. However, as this zero weighting modification makes successive calculations dependent upon previous ones, we need to compute Dijkstra's algorithm anew for each path.

### Non-Overlapped Chunked Processing (`fix_borders=True`)

When processing large volumes, a sensible approach for mass producing skeletons is to chunk the volume, process the chunks independently, and merge the resulting skeleton fragments at the end. However, this is complicated by the "edge effect" induced by a loss of context which makes it impossible to expect the endpoints of skeleton fragments produced by adjacent chunks to align. In contrast, it is easy to join mesh fragments because the vertices of the edge of mesh fragments lie at predictable identical locations given one pixel of overlap.  

Previously, we had used 50% overlap to join adjacent skeleton fragments which increased the compute cost of skeletonizing a large volume by eight times. However, if we could force skeletons to lie at predictable locations on the border, we could use single pixel overlap and copy the simple mesh joining approach. As an (incorrect but useful) intuition for how one might go about this, consider computing the centroid of each connected component on each border plane and adding that as a required path target. This would guarantee that both sides of the plane connect at the same pixel. However, the centroid may not lie inside of non-convex hulls so we have to be more sophisticated and select some real point inside of the shape.

To this end, we again repurpose the euclidean distance transform and apply it to each of the six planes of connected components and select the maximum value as a mandatory target. This works well for many types of objects that contact a single plane and have a single maximum. However, we must treat the corners of the box and shapes that have multiple maxima.  

To handle shapes that contact multiple sides of the box, we simply assign targets to all connected components. If this introduces a cycle in post-processing, we already have cycle removing code to handle it in Igneous. If it introduces tiny useless appendages, we also have code to handle this.  

If a shape has multiple distance transform maxima, it is important to choose the same pixel without needing to communicate between spatially adjacent tasks which may run at different times on different machines. Additionally, the same plane on adjacent tasks has the coordinate system flipped. One simple approach might be to pick the coordinate with minimum x and y (or some other coordinate based criterion) in one of the coordinate frames, but this requires tracking the flips on all six planes and is annoying. Instead, we use a series of coordinate-free topology based filters which is both more fun, effort efficient, and picks something reasonable looking. A valid criticism of this approach is that it will fail on a perfectly symmetrical object, but these objects are rare in biological data.  

We apply a series of filters and pick the point based on the first filter it passes:

1. The voxel closest to the centroid of the current label.
2. The voxel closest to the centroid of the image plane.
3. Closest to a corner of the plane.
4. Closest to an edge of the plane.
5. The previously found maxima.

It is important that filter #1 be based on the shape of the label so that kinks are minimimized for convex hulls. For example, originally we used only filters two thru five, but this caused skeletons for neurites located away from the center of a chunk to suddenly jink towards the center of the chunk at chunk boundaries.

### Rolling Invalidation Cube

The original TEASAR paper calls for a "rolling invalidation ball" that erases foreground voxels in step 6(iii). A naive implementation of this ball is very expensive as each voxel in the path requires its own ball, and many of these voxels overlap. In some cases, it is possible that the whole volume will need to be pointlessly reevaluated for every voxel along the path from root to target. While it's possible to special case the worst case, in the more common general case, a large amount of duplicate effort is expended.

Therefore, we applied an algorithm using topological cues to perform the invalidation operation in linear time. For simplicity of implmentation, we substituted a cube shape instead of a sphere. The function name `roll_invalidation_cube` is intended to evoke this awkwardness, though it hasn't appeared to have been  important.  

The two-pass algorithm is as follows. Given a binary image *I*, a skeleton *S*, and a set of vertices *V*:

1. Let *B<sub>v</sub>* be the set of bounding boxes that inscribe the spheres indicated by the TEASAR paper.
2. Allocate a 3D signed integer array, *T*, the size and dimension of *I* representing the topology. *T* is initially set to all zeros.
3. For each *B<sub>v</sub>*:
  1. Set T(p) += 1 for all points *p* on *B<sub>v</sub>*'s left boundary along the x-axis.
  2. Set T(p) -= 1 for all points *p* on *B<sub>v</sub>*'s right boundary along the x-axis.
4. Compute the bounding box *B<sub>global</sub>* that inscribes the union of all *B<sub>v</sub>*.
5. A point *p* travels along the x-axis for each row of *B<sub>global</sub>* starting on the YZ plane. 
  1. Set integer *coloring* = 0
  2. At each index, *coloring* += *T*(p)
  3. If *coloring* > 0 or *T*(p) is non-zero (we're on the leaving edge), we are inside an invalidation cube and start converting foreground voxels into background voxels.

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

Alex Bae developed the precursor skeletonization package and several modifications to TEASAR that we use in this package. Alex also developed the postprocessing approach used for stitching skeletons using 50% overlap. Will Silversmith adapted these techniques for mass production, refined several basic algorithms for handling thousands of labels at once, and rewrote them into the Kimimaro package. Will added trickle DAF, zero weighted previously explored paths, and fixing borders to the algorithm. Forrest Collman added parameter flexibility and helped tune DAF computation performance. Sven Dorkenwald and Forrest both provided helpful discussions and feedback.

## References 

1. M. Sato, I. Bitter, M.A. Bender, A.E. Kaufman, and M. Nakajima. "TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust Skeletons". Proc. 8th Pacific Conf. on Computer Graphics and Applications. Oct. 2000. doi: 10.1109/PCCGA.2000.883951 ([link](https://ieeexplore.ieee.org/abstract/document/883951/))
2. I. Bitter, A.E. Kaufman, and M. Sato. "Penalized-distance volumetric skeleton algorithm". IEEE Transactions on Visualization and Computer Graphics Vol. 7, Iss. 3, Jul-Sep 2001. doi: 10.1109/2945.942688 ([link](https://ieeexplore.ieee.org/abstract/document/942688/))
3. T. Zhao, S. Plaza. "Automatic Neuron Type Identification by Neurite Localization in the Drosophila Medulla". Sept. 2014. arXiv:1409.1892 \[q-bio.NC\] ([link](https://arxiv.org/abs/1409.1892))
4. A. Tagliasacchi, T. Delame, M. Spagnuolo, N. Amenta, A. Telea. "3D Skeletons: A State-of-the-Art Report". May 2016. Computer Graphics Forum. Vol. 35, Iss. 2. doi: 10.1111/cgf.12865 ([link](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12865))
5. P. Li, L. Lindsey, M. Januszewski, Z. Zheng, A. Bates, I. Taisz, M. Tyka, M. Nichols, F. Li, E. Perlman, J. Maitin-Shepard, T. Blakely, L. Leavitt, G. Jefferis, D. Bock, V. Jain. "Automated Reconstruction of a Serial-Section EM ​ Drosophila ​ Brain with Flood-Filling Networks and Local Realignment". April 2019. bioRXiv. doi: 10.1101/605634 ([link](https://www.biorxiv.org/content/10.1101/605634v1))
