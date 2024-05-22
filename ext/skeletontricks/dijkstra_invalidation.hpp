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
 * This algorithm is derived from dijkstra3d: 
 * https://github.com/seung-lab/dijkstra3d
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: May 2024
 */

#ifndef DIJKSTRA_INVALIDATION_HPP
#define DIJKSTRA_INVALIDATION_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

#include "./libdivide.h"

#define NHOOD_SIZE 26

namespace dijkstra_invalidation {

// helper function to compute 2D anisotropy ("_s" = "square")
inline float _s(const float wa, const float wb) {
  return std::sqrt(wa * wa + wb * wb);
}

// helper function to compute 3D anisotropy ("_c" = "cube")
inline float _c(const float wa, const float wb, const float wc) {
  return std::sqrt(wa * wa + wb * wb + wc * wc);
}

void connectivity_check(int connectivity) {
  if (connectivity != 6 && connectivity != 18 && connectivity != 26) {
    throw std::runtime_error("Only 6, 18, and 26 connectivities are supported.");
  }
}

void compute_neighborhood_helper_6(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz
) {

  const int sxy = sx * sy;

  // 6-hood
  neighborhood[0] = -1 * (x > 0); // -x
  neighborhood[1] = (x < (static_cast<int>(sx) - 1)); // +x
  neighborhood[2] = -static_cast<int>(sx) * (y > 0); // -y
  neighborhood[3] = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
  neighborhood[4] = -sxy * static_cast<int>(z > 0); // -z
  neighborhood[5] = sxy * (z < static_cast<int>(sz) - 1); // +z
}

void compute_neighborhood_helper_18(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
  // 6-hood
  compute_neighborhood_helper_6(neighborhood, x,y,z, sx,sy,sz);

  // 18-hood

  // xy diagonals
  neighborhood[6] = (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[7] = (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
  neighborhood[8] = (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
  neighborhood[9] = (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

  // yz diagonals
  neighborhood[10] = (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
  neighborhood[11] = (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
  neighborhood[12] = (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
  neighborhood[13] = (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

  // xz diagonals
  neighborhood[14] = (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
  neighborhood[15] = (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
  neighborhood[16] = (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
  neighborhood[17] = (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right
}

void compute_neighborhood_helper_26(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
  compute_neighborhood_helper_18(neighborhood, x,y,z, sx,sy,sz);
  
  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] = (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26, const uint32_t* voxel_connectivity_graph = NULL) {

  if (connectivity == 26) {
    compute_neighborhood_helper_26(neighborhood, x, y, z, sx, sy, sz);
  }
  else if (connectivity == 18) {
    compute_neighborhood_helper_18(neighborhood, x, y, z, sx, sy, sz);
  }
  else {
    compute_neighborhood_helper_6(neighborhood, x, y, z, sx, sy, sz);
  }

  if (voxel_connectivity_graph == NULL) {
    return;
  }

  uint64_t loc = x + sx * (y + sy * z);
  uint32_t graph = voxel_connectivity_graph[loc];

  // graph conventions are defined here:
  // https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92

  // 6-hood
  neighborhood[0] *= ((graph & 0b000010) > 0); // -x
  neighborhood[1] *= ((graph & 0b000001) > 0); // +x
  neighborhood[2] *= ((graph & 0b001000) > 0); // -y
  neighborhood[3] *= ((graph & 0b000100) > 0); // +y
  neighborhood[4] *= ((graph & 0b100000) > 0); // -z
  neighborhood[5] *= ((graph & 0b010000) > 0); // +z

  // 18-hood

  // xy diagonals
  neighborhood[6] *= ((graph & 0b1000000000) > 0); // up-left -x,-y
  neighborhood[7] *= ((graph & 0b0010000000) > 0); // up-right -x,+y
  neighborhood[8] *= ((graph & 0b0100000000) > 0); // down-left +x,-y
  neighborhood[9] *= ((graph & 0b0001000000) > 0); // down-right +x,+y

  // yz diagonals
  neighborhood[10] *= ((graph & 0b100000000000000000) > 0); // up-left -y,-z
  neighborhood[11] *= ((graph & 0b000010000000000000) > 0); // up-right -y,+z
  neighborhood[12] *= ((graph & 0b010000000000000000) > 0); // down-left +y,-z
  neighborhood[13] *= ((graph & 0b000001000000000000) > 0); // down-right +y,+z

  // xz diagonals
  neighborhood[14] *= ((graph & 0b001000000000000000) > 0); // up-left, -x,-z
  neighborhood[15] *= ((graph & 0b000000100000000000) > 0); // up-right, -x,+z
  neighborhood[16] *= ((graph & 0b000100000000000000) > 0); // down-left +x,-z
  neighborhood[17] *= ((graph & 0b000000010000000000) > 0); // down-right +x,+z

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] *= ((graph & 0b10000000000000000000000000) > 0); // -x,-y,-z
  neighborhood[19] *= ((graph & 0b01000000000000000000000000) > 0); // +x,-y,-z
  neighborhood[20] *= ((graph & 0b00100000000000000000000000) > 0); // -x,+y,-z
  neighborhood[21] *= ((graph & 0b00001000000000000000000000) > 0); // -x,-y,+z
  neighborhood[22] *= ((graph & 0b00010000000000000000000000) > 0); // +x,+y,-z
  neighborhood[23] *= ((graph & 0b00000100000000000000000000) > 0); // +x,-y,+z
  neighborhood[24] *= ((graph & 0b00000010000000000000000000) > 0); // -x,+y,+z
  neighborhood[25] *= ((graph & 0b00000001000000000000000000) > 0); // +x,+y,+z
}

#define DIJKSTRA_3D_PREFETCH_26WAY(field, loc) \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sxy - sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sxy - sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) + sx - 1]), 0, 1); \
  HEDLEYX_PREFETCH(reinterpret_cast<char*>(&field[(loc) - sx - 1]), 0, 1);

class HeapDistanceNode {
public:
  float dist;
  uint64_t original_loc;
  uint64_t value;
  float max_dist;

  HeapDistanceNode() {
    dist = 0;
    value = 0;
    original_loc = 0;
    max_dist = 0;
  }

  HeapDistanceNode (float d, uint64_t o_loc, uint64_t val, float mx_dist) {
    dist = d;
    value = val;
    original_loc = o_loc;
    max_dist = mx_dist;
  }

  HeapDistanceNode (const HeapDistanceNode &h) {
    dist = h.dist;
    value = h.value;
    max_dist = h.max_dist;
    original_loc = h.original_loc;
  }
};

struct HeapDistanceNodeCompare {
  bool operator()(const HeapDistanceNode &t1, const HeapDistanceNode &t2) const {
    return t1.dist >= t2.dist;
  }
};

int64_t _roll_invalidation_ball(
  uint8_t* field, // really a boolean field
  const uint64_t sx, const uint64_t sy, const uint64_t sz, 
  const float wx, const float wy, const float wz, 
  const std::vector<uint64_t> &sources,
  const std::vector<float> &max_distances,
  const int connectivity = 26, 
  const uint32_t* voxel_connectivity_graph = NULL
) {

  const uint64_t sxy = sx * sy;

  const libdivide::divider<uint64_t> fast_sx(sx); 
  const libdivide::divider<uint64_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  connectivity_check(connectivity);

  int neighborhood[NHOOD_SIZE] = {};

  std::priority_queue<
    HeapDistanceNode, std::vector<HeapDistanceNode>, HeapDistanceNodeCompare
  > queue;

  for (uint64_t i = 0; i < sources.size(); i++) {
    queue.emplace(0.0, sources[i], sources[i], max_distances[i]);
  }

  uint64_t loc;
  uint64_t neighboridx;

  int64_t x, y, z;
  int64_t orig_x, orig_y, orig_z;

  int64_t invalidated = 0;

  auto xyzfn = [=](uint64_t l, int64_t& x, int64_t& y, int64_t& z) {
    if (power_of_two) {
      z = l >> (xshift + yshift);
      y = (l - (z << (xshift + yshift))) >> xshift;
      x = l - ((y + (z << yshift)) << xshift);
    }
    else {
      z = l / fast_sxy;
      y = (l - (z * sxy)) / fast_sx;
      x = l - sx * (y + z * sy);
    }
  };

  while (!queue.empty()) {
    const float max_dist = queue.top().max_dist;
    const uint64_t original_loc = queue.top().original_loc;
    loc = queue.top().value;
    queue.pop();

    if (!field[loc]) {
      continue;
    }

    field[loc] = 0;
    invalidated++;

    xyzfn(loc, x, y, z);
    xyzfn(original_loc, orig_x, orig_y, orig_z);
    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity, voxel_connectivity_graph);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      if (field[neighboridx] == 0) {
        continue;
      }

      xyzfn(neighboridx, x, y, z);
      float new_dist = _c(
        wx * static_cast<float>(x - orig_x), 
        wy * static_cast<float>(y - orig_y), 
        wz * static_cast<float>(z - orig_z)
      );

      if (new_dist < max_dist) { 
        queue.emplace(new_dist, original_loc, neighboridx, max_dist);
      }
    }
  }

  return invalidated;
}

};

#undef NHOOD_SIZE
#undef DIJKSTRA_3D_PREFETCH_26WAY

#endif
