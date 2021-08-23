import os

from cloudvolume.lib import mkdir
import click
import numpy as np

import kimimaro

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
@click.option('--pdrf-scale', type=float, default=1e5, help="Constant multiplier of penalty field.", show_default=True)
@click.option('--pdrf-exponent', type=float, default=4, help="Exponent of penalty field. Powers of two are faster. Too big can cause floating point errors.", show_default=True)
@click.option('--soma-detect', type=float, default=None, help="If specified, distance to boundary values above this threshold trigger special soma processing. e.g. 750 nm", show_default=True)
@click.option('--soma-scale', type=float, default=2, help="Adds multiple of boundary distance to invalidation zone around a soma. (You should set this!)", show_default=True)
@click.option('--soma-const', type=float, default=300, help="Adds constant physical distance to invalidation zone around a soma. (You should set this!)", show_default=True)
@click.option('--anisotropy', type=Tuple3(), default="1,1,1", help="Physical size of voxel in x,y,z axes.", show_default=True)
@click.option('--dust', type=int, default=1000, help="Skip connected components with fewer voxels than this.", show_default=True)
@click.option('--progress', is_flag=True, default=False, help="Show progress bar.", show_default=True)
@click.option('--fill-holes/--no-fill-holes', is_flag=True, default=True, help="Fill holes in each connected component. (slower)", show_default=True)
@click.option('--fix-avocados', is_flag=True, default=False, help="Use heuristics to combine nucleii with cell bodies. (slower)", show_default=True)
@click.option('--fix-borders', is_flag=True, default=False, help="Center the skeleton where the shape contacts the border.", show_default=True)
@click.option('--fix-branches', is_flag=True, default=True, help="Improves quality of forked shapes. (slower for highly branched shapes)", show_default=True)
@click.option('--parallel', type=int, default=1, help="Number of processes to use.", show_default=True)
@click.option('--outdir', type=str, default="kimimaro_out", help="Where to write the SWC files.", show_default=True)
def forge(
  src,
  scale, const, pdrf_scale, pdrf_exponent,
  soma_detect, soma_scale, soma_const,
  anisotropy, dust, progress, fill_holes, 
  fix_avocados, fix_branching, fix_borders,
  parallel, outdir
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
      "soma_invalidation_scale": soma_scale,
      "soma_invalidation_const": soma_const,
    },
    anisotropy=anisotropy,
    dust_threshold=dust,
    progress=progress,
    fill_holes=fill_holes,
    fix_avocados=fix_avocados,
    fix_branching=fix_branching,
    fix_borders=fix_borders,
    parallel=parallel,
  )

  directory = mkdir(outdir)

  for label, skel in skels.items():
    fname = os.path.join(directory, f"{label}.swc")
    with open(fname, "wb") as f:
      f.write(skel.to_swc())

  if progress:
    print(f"kimimaro: wrote {len(skels)} skeletons to {directory}")

@main.command()
def license():
  """Prints the license for this library and cli tool."""
  path = os.path.join(os.path.dirname(__file__), 'LICENSE')
  with open(path, 'rt') as f:
    print(f.read())






