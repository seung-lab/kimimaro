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
#include <stack>
#include <unordered_map>
#include <string>
#include <set>

#include "libdivide.h"

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

  std::vector< std::set<T> > index(Nv);
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

void _binary_fill_holes(
  uint8_t* labels, 
  const size_t sx, const size_t sy, const size_t sz
) {

  const size_t sxy = sx * sy;
  const size_t voxels = sx * sy * sz;

  if (voxels == 0) {
    return;
  }

  const size_t sxv = sx + 2;
  const size_t syv = sy + 2;
  const size_t szv = sz + 2;
  const size_t sxyv = sxv * syv;

  uint8_t* visited = new uint8_t[sxyv * szv](); 

  // paint labels into visited offset by +<1,1,1>
  // and mark all foreground as 2 so we can mark
  // visited as 1 without overwriting foreground
  // as we want foreground to be 2 and voids to 
  // be 0
  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      for (size_t x = 0; x < sx; x++) {
        size_t i = x + sx * y + sxy * z;
        visited[(x+1) + sxv * (y+1) + sxyv * (z+1)] = static_cast<uint8_t>(labels[i] > 0) << 1;
      }
    }
  }

  const libdivide::divider<size_t> fast_sxv(sxv); 
  const libdivide::divider<size_t> fast_sxyv(sxyv); 

  std::stack<size_t> stack;
  stack.push(0);

  while (!stack.empty()) {
    size_t loc = stack.top();
    stack.pop();

    if (visited[loc]) {
      continue;
    }

    size_t z = loc / fast_sxyv;
    size_t y = (loc - (z * sxyv)) / fast_sxv;
    size_t x = loc - sxv * (y + z * syv);

    visited[loc] = 1;

    if (x > 0 && !visited[loc-1]) {
      stack.push( loc - 1 );
    }
    if (x < sxv - 1 && !visited[loc+1]) {
      stack.push( loc + 1 );
    }
    if (y > 0 && !visited[loc-sxv]) {
      stack.push( loc - sxv );
    }
    if (y < syv - 1 && !visited[loc+sxv]) {
      stack.push( loc + sxv );
    }
    if (z > 0 && !visited[loc-sxyv]) {
      stack.push( loc - sxyv );
    }
    if (z < szv - 1 && !visited[loc+sxyv]) {
      stack.push( loc + sxyv );
    }
  }

  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      for (size_t x = 0; x < sx; x++) {
        labels[ x + sx * y + sxy * z ] = static_cast<uint8_t>(
          visited[ (x+1) + sxv * (y+1) + sxyv * (z+1) ] != 1
        );
      }
    }
  }

  delete[] visited;
}


};

#endif