"""
Optional GPU backend for kimimaro.skeletonize(..., gpu=True).

The work is delegated to Brook (pip package brook-cu12), a CUDA
implementation of Kimimaro's TEASAR variant. Brook is imported only
when the GPU backend is used, never at import time of this module.

This file is part of Kimimaro.

Kimimaro is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Kimimaro is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Kimimaro.  If not, see <https://www.gnu.org/licenses/>.
"""

import warnings

def _import_brook():
  try:
    import brook
  except ImportError as err:
    raise ImportError(
      "kimimaro.skeletonize(..., gpu=True) requires the Brook GPU backend. "
      "Install it with: pip install \"kimimaro[gpu]\" "
      "(requires Linux x86-64, Python 3.12 or newer and an NVIDIA GPU "
      "of compute capability 8.0 or newer)."
    ) from err

  wrong_package = ImportError(
    "The installed 'brook' module ({}) is not the Brook GPU skeletonizer. "
    "Uninstall it and install brook-cu12 instead: pip install \"kimimaro[gpu]\"".format(
      getattr(brook, "__file__", None) or "unknown location"
    )
  )

  if not hasattr(brook, "skeletonize"):
    raise wrong_package

  try:
    from brook import device # "import brook" alone does not load brook.device
  except ImportError as err:
    raise wrong_package from err

  return brook

def is_available():
  """
  Returns True if kimimaro.skeletonize(..., gpu=True) can run here:
  Brook is installed and a supported NVIDIA GPU is usable. Never raises.

  This initializes CUDA in the calling process (about 0.2 s and a few
  hundred MB of GPU memory held until the process exits), so call it once.
  Processes started with parallel > 1 re-import __main__, so guard the
  call with if __name__ == "__main__".
  """
  try:
    brook = _import_brook()
    if brook.device.device_count() <= 0:
      return False
    brook.device.require_gpu()
  except Exception:
    return False
  return True

def skeletonize(
  all_labels, teasar_params, anisotropy,
  object_ids, dust_threshold,
  progress, fix_branching, in_place,
  fix_borders, parallel, parallel_chunk_size,
  extra_targets_before, extra_targets_after,
  fill_holes, fix_avocados,
  voxel_graph
):
  """
  GPU implementation of kimimaro.skeletonize. Use it via
  kimimaro.skeletonize(..., gpu=True), which supplies the defaults.

  parallel and parallel_chunk_size are ignored (a warning is
  emitted if parallel != 1).

  Returns: { $segid: osteoid.Skeleton, ... }
  """
  brook = _import_brook()
  brook.device.require_gpu()

  if parallel != 1:
    warnings.warn(
      "parallel={} is ignored when gpu=True; Brook skeletonizes on a single GPU.".format(parallel),
      UserWarning, stacklevel=3
    )

  kwargs = {}
  if voxel_graph is not None or fill_holes or fix_avocados:
    # Brook streams volumes it estimates will not fit in GPU memory,
    # but its streaming mode raises NotImplementedError for these options.
    kwargs["streaming"] = False

  try:
    return brook.skeletonize(
      all_labels,
      teasar_params=teasar_params,
      anisotropy=anisotropy,
      object_ids=object_ids,
      dust_threshold=dust_threshold,
      progress=progress,
      fix_branching=fix_branching,
      in_place=in_place,
      fix_borders=fix_borders,
      extra_targets_before=extra_targets_before,
      extra_targets_after=extra_targets_after,
      fill_holes=fill_holes,
      fix_avocados=fix_avocados,
      voxel_graph=voxel_graph,
      **kwargs
    )
  except brook.DimensionError as err:
    from .intake import DimensionError
    raise DimensionError(str(err)) from err
