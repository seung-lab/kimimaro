import os

import microviewer
import click
import numpy as np
from osteoid import Skeleton

import kimimaro
from kimimaro.utility import mkdir
import fastremap
from tqdm import tqdm

class Tuple3(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple3'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 3:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value


@click.group()
def main():
  """
  Skeletonize all labels in a segmented volumetric image
  by applying a TEASAR based algorithm and outputs them
  as SWC.

  Does not accept continuously valued images such as raw
  microscopy images.

  Input File Formats Supported: npy
  
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version. Run "igneous license" for details.  
  """
  pass

@main.command()
@click.argument("src")
@click.option('--scale', type=float, default=4, help="Adds multiple of boundary distance to invalidation zone. (You should set this!)", show_default=True)
@click.option('--const', type=float, default=10, help="Adds constant physical distance to invalidation zone. (You should set this!)", show_default=True)
@click.option('--pdrf-scale', type=int, default=1e5, help="Constant multiplier of penalty field.", show_default=True)
@click.option('--pdrf-exponent', type=int, default=4, help="Exponent of penalty field. Powers of two are faster. Too big can cause floating point errors.", show_default=True)
@click.option('--soma-detect', type=float, default=750, help="Perform more expensive check for somas for distance to boundary values above this threshold. e.g. 750 nm", show_default=True)
@click.option('--soma-accept', type=float, default=1100, help="Distance to boundary values above this threshold trigger special soma processing. e.g. 750 nm", show_default=True)
@click.option('--soma-scale', type=float, default=2, help="Adds multiple of boundary distance to invalidation zone around a soma. (You should set this!)", show_default=True)
@click.option('--soma-const', type=float, default=300, help="Adds constant physical distance to invalidation zone around a soma. (You should set this!)", show_default=True)
@click.option('--anisotropy', type=Tuple3(), default="1,1,1", help="Physical size of voxel in x,y,z axes.", show_default=True)
@click.option('--dust', type=int, default=1000, help="Skip connected components with fewer voxels than this.", show_default=True)
@click.option('--progress', is_flag=True, default=False, help="Show progress bar.", show_default=True)
@click.option('--fill-holes/--no-fill-holes', is_flag=True, default=True, help="Fill holes in each connected component. (slower)", show_default=True)
@click.option('--fix-avocados', is_flag=True, default=False, help="Use heuristics to combine nucleii with cell bodies. (slower)", show_default=True)
@click.option('--fix-borders', is_flag=True, default=False, help="Center the skeleton where the shape contacts the border.", show_default=True)
@click.option('--fix-branches', is_flag=True, default=True, help="Improves quality of forked shapes. (slower for highly branched shapes)", show_default=True)
@click.option('--max-paths', type=int, default=None, help="Maximum number of paths to trace per object.", show_default=True)
@click.option('-p', '--parallel', type=int, default=1, help="Number of processes to use.", show_default=True)
@click.option('-o', '--outdir', type=str, default="kimimaro_out", help="Where to write the SWC files.", show_default=True)
def forge(
  src,
  scale, const, pdrf_scale, pdrf_exponent,
  soma_detect, soma_accept, soma_scale, soma_const,
  anisotropy, dust, progress, fill_holes, 
  fix_avocados, fix_branches, fix_borders,
  parallel, max_paths, outdir
):
  """Skeletonize an input image and write out SWCs."""
  
  labels = np.load(src)

  skels = kimimaro.skeletonize(
    labels,
    teasar_params={
      "scale": scale,
      "const": const,
      "pdrf_scale": pdrf_scale,
      "pdrf_exponent": pdrf_exponent,
      "soma_detection_threshold": soma_detect,
      "soma_acceptance_threshold": soma_accept,
      "soma_invalidation_scale": soma_scale,
      "soma_invalidation_const": soma_const,
      "max_paths": max_paths,
    },
    anisotropy=anisotropy,
    dust_threshold=dust,
    progress=progress,
    fill_holes=fill_holes,
    fix_avocados=fix_avocados,
    fix_branching=fix_branches,
    fix_borders=fix_borders,
    parallel=parallel,
  )

  directory = mkdir(outdir)

  for label, skel in skels.items():
    fname = os.path.join(directory, f"{label}.swc")
    with open(fname, "wt") as f:
      f.write(skel.to_swc())

  if progress:
    print(f"kimimaro: wrote {len(skels)} skeletons to {directory}")

@main.group()
def swc():
  """Utilities for managing SWC files. Use forge to create new skeletons."""
  pass

@swc.command("from")
@click.argument("src", nargs=-1)
def from_image(src):
  """Convert a binary image that has already been skeletonized by a thinning algorithm into an swc."""

  for srcpath in tqdm(src):
    basename, ext = os.path.splitext(srcpath)
    if ext == ".npy":
      image = np.load(srcpath)
    elif ext in (".tif", ".tiff"):
      try:
        import tifffile
      except ImportError:
        print("kimimaro: tifffile not installed. Run pip install tifffile.")
        return
      image = tifffile.imread(srcpath)
    else:
      print(f"Unsupported image format {ext}. Only npy and tiff are supported.")
      return

    image = np.asfortranarray(image)
    skel = kimimaro.extract_skeleton_from_binary_image(image)

    with open(f"{basename}.swc", "wt") as f:
      f.write(skel.to_swc())

@swc.command("to")
@click.argument("src", nargs=-1)
@click.option('--format', type=str, default="npy", help="Which format to use. Options: npy, tiff", show_default=True)
def to_image(src, format):
  """Convert an swc into a binary image."""
  if format not in ("npy", "tiff"):
    print(f"kimimaro: invalid format {format}. npy or tiff allowed.")

  for srcpath in tqdm(src):
    with open(srcpath, 'rt') as f:
      skel = Skeleton.from_swc(f.read())

    xmin, xmax = fastremap.minmax(skel.vertices[:,0])
    ymin, ymax = fastremap.minmax(skel.vertices[:,1])
    zmin, zmax = fastremap.minmax(skel.vertices[:,2])

    image = np.zeros((xmax-xmin, ymax-ymin, zmax-zmin), dtype=np.bool, order='F')
    minpt = np.array([xmin,ymin,zmin])
    drawpts = skel.vertices - minpt

    image[drawpts] = True

    basename, ext = os.path.splitext(srcpath)

    if format == "npy":
      np.save(f"{basename}.npy", image)
    elif format == "tiff":
      try:
        import tifffile
        tifffile.imwrite(f"{basename}.tiff", image, photometric='minisblack')
      except ImportError:
        print("kimimaro: tifffile not installed. Run pip install tifffile.")
        return
    else:
      raise ValueError("should never happen")

@main.command()
@click.argument("filename")
@click.option('--port', type=int, default=8080, help="Which port to run the microviewer on for npy files.", show_default=True)
def view(filename, port):
  """Visualize a .swc or .npy file."""
  basename, ext = os.path.splitext(filename)

  if ext == ".swc":
    with open(filename, "rt") as swc:
      skel = Skeleton.from_swc(swc.read())

    skel.viewer()
  elif ext == ".npy":
    labels = np.load(filename)
    microviewer.view(labels, segmentation=True, port=port)
  else:
    print("kimimaro: {filename} was not a .swc or .npy file.")

@main.command()
def license():
  """Prints the license for this library and cli tool."""
  path = os.path.join(os.path.dirname(__file__), 'LICENSE')
  with open(path, 'rt') as f:
    print(f.read())

