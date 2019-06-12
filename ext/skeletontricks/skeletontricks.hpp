/*
 * This file is part of Kimimaro.
 * 
 * Kimimaro is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Kimimaro is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Kimimaro.  If not, see <https://www.gnu.org/licenses/>.
 *
 * 
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: September 2018
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <vector>

#ifndef SKELETONTRICKS_HPP
#define SKELETONTRICKS_HPP

namespace skeletontricks {

int _roll_invalidation_cube(
  uint8_t* labels, float* DBF,
  const int sx, const int sy, const int sz,
  const float wx, const float wy, const float wz,
  int* path, const int path_size,
  const float scale, const float constant) {

  if (path_size == 0) {
    return 0;
  }

  const int sxy = sx * sy;
  const int voxels = sxy * sz;

  int minx, maxx, miny, maxy, minz, maxz;
  int x, y, z;

  int global_minx = sx;
  int global_maxx = 0;
  int global_miny = sy;
  int global_maxy = 0;
  int global_minz = sz;
  int global_maxz = 0;

  int16_t* topology = new int16_t[voxels]();
  
  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  int loc;
  float radius;

  // First pass: compute toplology
  for (int i = 0; i < path_size; i++) {
    loc = path[i];
    radius = scale * DBF[loc] + constant;

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / sxy;
      y = (loc - (z * sxy)) / sx;
      x = loc - sx * (y + z * sy);
    }

    minx = std::max(0,     (int)(x - (radius / wx)));
    maxx = std::min(sx-1,  (int)(0.5 + (x + (radius / wx))));
    miny = std::max(0,     (int)(y - (radius / wy)));
    maxy = std::min(sy-1,  (int)(0.5 + (y + (radius / wy))));
    minz = std::max(0,     (int)(z - (radius / wz)));
    maxz = std::min(sz-1,  (int)(0.5 + (z + (radius / wz))));

    global_minx = std::min(global_minx, minx);
    global_maxx = std::max(global_maxx, maxx);
    global_miny = std::min(global_miny, miny);
    global_maxy = std::max(global_maxy, maxy);
    global_minz = std::min(global_minz, minz);
    global_maxz = std::max(global_maxz, maxz);

    for (y = miny; y <= maxy; y++) {
      for (z = minz; z <= maxz; z++) {
        topology[minx + sx * y + sxy * z] += 1;
        topology[maxx + sx * y + sxy * z] -= 1;
      }
    }
  }

  // Second pass: invalidate labels
  int coloring;
  int invalidated = 0;
  int yzoffset;
  for (z = global_minz; z <= global_maxz; z++) {
    for (y = global_miny; y <= global_maxy; y++) {
      yzoffset = sx * y + sxy * z;

      coloring = 0;
      for (x = global_minx; x <= global_maxx; x++) {
        coloring += topology[x + yzoffset];
        if (coloring > 0 || topology[x + yzoffset]) {
          invalidated += static_cast<int>(labels[x + yzoffset] > 0); // convert non-bool vals
          labels[x + yzoffset] = 0;
        }
      }
    }
  }

  free(topology);

  return invalidated;
}

};

#endif