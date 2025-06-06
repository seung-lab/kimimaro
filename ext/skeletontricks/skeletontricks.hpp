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
 * Date: September 2018 - April 2025
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <utility>

#include "unordered_dense.hpp"

#ifndef SKELETONTRICKS_HPP
#define SKELETONTRICKS_HPP

namespace skeletontricks {

size_t _roll_invalidation_cube(
  uint8_t* labels, float* DBF,
  const int64_t sx, const int64_t sy, const int64_t sz,
  const float wx, const float wy, const float wz,
  size_t* path, const size_t path_size,
  const float scale, const float constant
) {

  if (path_size == 0) {
    return 0;
  }

  const size_t sxy = sx * sy;
  const size_t voxels = sxy * sz;

  int64_t minx, maxx, miny, maxy, minz, maxz;
  int64_t x, y, z;

  int64_t global_minx = sx;
  int64_t global_maxx = 0;
  int64_t global_miny = sy;
  int64_t global_maxy = 0;
  int64_t global_minz = sz;
  int64_t global_maxz = 0;

  std::vector<int16_t> topology(voxels);
  
  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  size_t loc;
  float radius;

  // First pass: compute toplology
  for (size_t i = 0; i < path_size; i++) {
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

    const int64_t ZERO = 0;

    minx = std::max(ZERO,    static_cast<int64_t>(x - (radius / wx)));
    maxx = std::min(sx-1, static_cast<int64_t>(0.5 + (x + (radius / wx))));
    miny = std::max(ZERO,    static_cast<int64_t>(y - (radius / wy)));
    maxy = std::min(sy-1, static_cast<int64_t>(0.5 + (y + (radius / wy))));
    minz = std::max(ZERO,    static_cast<int64_t>(z - (radius / wz)));
    maxz = std::min(sz-1, static_cast<int64_t>(0.5 + (z + (radius / wz))));

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
  size_t invalidated = 0;
  size_t yzoffset;
  for (z = global_minz; z <= global_maxz; z++) {
    for (y = global_miny; y <= global_maxy; y++) {
      yzoffset = sx * y + sxy * z;

      coloring = 0;
      for (x = global_minx; x <= global_maxx; x++) {
        coloring += topology[x + yzoffset];
        if (coloring > 0 || topology[x + yzoffset]) {
          invalidated += static_cast<size_t>(labels[x + yzoffset] > 0); // convert non-bool vals
          labels[x + yzoffset] = 0;
        }
      }
    }
  }

  return invalidated;
}

template <typename T>
inline size_t max(T* edges, const size_t size) {
  if (size == 0) {
    return 0;
  }

  size_t mx = edges[0];
  for (size_t i = 0; i < size; i++) {
    if (static_cast<size_t>(edges[i]) > mx) {
      mx = static_cast<size_t>(edges[i]);
    }
  }

  return mx;
}

template <typename T>
void printvec(std::vector<T> vec) {
  for (T v : vec) {
    printf("%d, ", v);
  }
  printf("\n");
}

template <typename T>
void printstack(std::stack<T> stack) {
  while (!stack.empty()) {
    printf("%d, ", stack.top());
    stack.pop();
  }

  printf("\n");
}

template <typename T>
std::vector<T> stack2vec(std::stack<T> stk) {
  std::vector<T> vec;
  vec.reserve(stk.size());

  while (!stk.empty()) {
    vec.push_back(stk.top());
    stk.pop();
  }

  std::reverse(vec.begin(), vec.end());

  return vec;
}

// Ne = size of edges / 2
// Nv = number of vertices (max of edge values)
template <typename T>
std::vector<T> _find_cycle(const T* edges, const size_t Ne) {
  if (Ne == 0) {
    return std::vector<T>(0);
  }

  size_t Nv = max(edges, Ne * 2) + 1; // +1 to ensure zero is counted

  std::vector< ankerl::unordered_dense::set<T> > index(Nv);
  index.reserve(Nv);

  // NB: consolidate handles the trivial loops (e1 == e2)
  //     and deduplication of edges
  for (size_t i = 0; i < 2 * Ne; i += 2) {
    T e1 = edges[i];
    T e2 = edges[i+1];

    index[e1].insert(e2);
    index[e2].insert(e1);
  }

  T root = edges[0];
  T node = -1;
  T parent = -1;
  uint32_t depth = -1;

  std::stack<T> stack;
  std::stack<T> parents;
  std::stack<uint32_t> depth_stack;
  std::stack<T> path;

  stack.push(root);
  parents.push(-1);
  depth_stack.push(0);
  
  std::vector<bool> visited(Nv, false);

  while (!stack.empty()) {
    node = stack.top();
    parent = parents.top();
    depth = depth_stack.top();

    stack.pop();
    parents.pop();
    depth_stack.pop();

    while (path.size() > depth) {
      path.pop();
    }

    path.push(node);

    if (visited[node]) {
      break;
    }
    visited[node] = true;

    for (T child : index[node]) {
      if (child == parent) {
        continue;
      }

      stack.push(child);
      parents.push(node);
      depth_stack.push(depth + 1);
    }
  }

  if (path.size() <= 1) {
    return std::vector<T>(0);
  }

  // cast stack to vector w/ zero copy
  std::vector<T> vec_path = stack2vec<T>(path);

  // Find start of loop. Since a cycle was detected,
  // the last node found started the cycle. We need
  // to trim the path leading up to that connection.
  size_t i;
  for (i = 0; i < vec_path.size() - 1; i++) {
    if (vec_path[i] == node) {
      break;
    }
  }

  if (vec_path.size() - i < 3) {
    return std::vector<T>(0);
  }

  return std::vector<T>(vec_path.begin() + i, vec_path.end());
}

// Had trouble returning an unordered_map< pair<int,int>, float>
// to python, so I decided to just pack two uint32s into a uint64
// and unpack them on the other side.
std::unordered_map<uint64_t, float> _create_distance_graph(
  float* vertices, size_t Nv, 
  uint32_t* edges, size_t Ne, uint32_t start_node,
  std::vector<int32_t> critical_points_vec
) {

  std::vector< std::vector<uint32_t> > tree(Nv);
  tree.reserve(Nv);

  std::vector<bool> critical_points(Nv, false);
  for (uint32_t edge : critical_points_vec) {
    critical_points[edge] = true;
  }

  for (size_t i = 0; i < Ne; i++) {
    uint32_t e1 = edges[2*i];
    uint32_t e2 = edges[2*i + 1];

    tree[e1].push_back(e2);
    tree[e2].push_back(e1);
  }

  std::unordered_map<uint64_t, float> distgraph;

  std::stack<uint32_t> stack;
  std::stack<int32_t> parents;
  std::stack<float> dist_stack;
  std::stack<uint32_t> root_stack;

  stack.push(start_node);
  parents.push(-1);
  dist_stack.push(0.0);
  root_stack.push(start_node);

  uint32_t node, root;
  int32_t parent;
  float dist;

  uint64_t key = 0;

  std::vector<bool> visited(Nv, false);

  while (!stack.empty()) {
    node = stack.top();
    dist = dist_stack.top();
    root = root_stack.top();
    parent = parents.top();

    if (visited[node]) {
      throw std::runtime_error(std::string("Cycle detected. Node: ") + std::to_string(node));
    }
    visited[node] = true;

    stack.pop();
    dist_stack.pop();
    root_stack.pop();
    parents.pop();

    if (critical_points[node] && node != root) {
      key = (root < node)
        ? static_cast<uint64_t>(root) | (static_cast<uint64_t>(node) << 32)
        : static_cast<uint64_t>(node) | (static_cast<uint64_t>(root) << 32);

      distgraph[key] = dist;
      dist = 0.0;
      root = node;
    }

    for (int32_t child : tree[node]) {
      if (static_cast<int32_t>(child) == parent) {
        continue;
      }

      float dx = vertices[3*node + 0] - vertices[3*child + 0];
      float dy = vertices[3*node + 1] - vertices[3*child + 1];
      float dz = vertices[3*node + 2] - vertices[3*child + 2];

      dx *= dx;
      dy *= dy;
      dz *= dz;

      stack.push(child);
      parents.push(static_cast<int32_t>(node));
      dist_stack.push(
        dist + sqrt(dx + dy + dz)
      );
      root_stack.push(root);
    }
  }

  return distgraph;
}

// extracting skeletons from binary images produced by
// other thinning based skeletonization algorithms

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26
) {

  const int sxy = sx * sy;

  const int plus_x = (x < (static_cast<int>(sx) - 1)); // +x
  const int minus_x = -1 * (x > 0); // -x
  const int plus_y = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
  const int minus_y = -static_cast<int>(sx) * (y > 0); // -y
  const int minus_z = -sxy * static_cast<int>(z > 0); // -z

  // 6-hood
  neighborhood[0] = minus_x;
  neighborhood[1] = minus_y;
  neighborhood[2] = minus_z;
  
  // 18-hood

  // xy diagonals
  neighborhood[3] = (connectivity > 6) * (minus_x + minus_y) * (minus_x && minus_y); // up-left
  neighborhood[4] = (connectivity > 6) * (plus_x + minus_y) * (plus_x && minus_y); // up-right

  // yz diagonals
  neighborhood[5] = (connectivity > 6) * (minus_x + minus_z) * (minus_x && minus_z); // down-left
  neighborhood[6] = (connectivity > 6) * (plus_x + minus_z) * (plus_x && minus_z); // down-right

  // xz diagonals
  neighborhood[7] = (connectivity > 6) * (minus_y + minus_z) * (minus_y && minus_z); // down-left
  neighborhood[8] = (connectivity > 6) * (plus_y + minus_z) * (plus_y && minus_z); // down-right

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[9] = (connectivity > 18) * (minus_x + minus_y + minus_z) * (minus_y && minus_z);
  neighborhood[10] = (connectivity > 18) * (plus_x + minus_y + minus_z) * (minus_y && minus_z);
  neighborhood[11] = (connectivity > 18) * (minus_x + plus_y + minus_z) * (plus_y && minus_z);
  neighborhood[12] = (connectivity > 18) * (plus_x + plus_y + minus_z) * (plus_y && minus_z);
}

struct pair_hash {
  inline std::size_t operator()(const std::pair<uint64_t,uint64_t> & v) const {
    return v.first * 31 + v.second; // arbitrary hash fn
  }
};

std::unordered_set<std::pair<uint64_t, uint64_t>, pair_hash> 
_extract_edges_from_binary_image(
  const uint8_t* image,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26
) {

  const uint64_t sxy = sx * sy;

  std::unordered_set<std::pair<uint64_t, uint64_t>, pair_hash> edges;
  edges.reserve(sx * sy * sz / 100);

  int neighborhood[13];
  uint64_t neighboridx = 0;

  for (uint64_t z = 0; z < sz; z++) {
    for (uint64_t y = 0; y < sy; y++) {
      for (uint64_t x = 0; x < sx; x++) {
        uint64_t loc = x + sx * y + sxy * z;
        if (image[loc] == 0) {
          continue;
        }

        compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

        for (int i = 0; i < 13; i++) {
          if (neighborhood[i] == 0) {
            continue;
          }

          neighboridx = loc + neighborhood[i];
          if (image[neighboridx] == 0) {
            continue;
          }

          if (loc <= neighboridx) {
            edges.emplace(std::make_pair(loc, neighboridx));
          } 
          else {
            edges.emplace(std::make_pair(neighboridx, loc));
          }
        }
      }
    }
  }

  return edges;
}

};

#endif
