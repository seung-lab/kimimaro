import numpy as np
from cloudvolume import Skeleton
import kimimaro.skeletontricks

def extract_skeleton_from_binary_image(image):
	verts, edges = kimimaro.skeletontricks.extract_edges_from_binary_image(image)
	return Skeleton(verts, edges)

