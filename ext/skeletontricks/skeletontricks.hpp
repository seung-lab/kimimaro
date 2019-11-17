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

  std::vector< std::vector<T> > index(Nv);
  index.reserve(Nv);

  // NB: consolidate handles the trivial loops (e1 == e2)
  //     and deduplication of edges
  for (size_t i = 0; i < 2 * Ne; i += 2) {
    T e1 = edges[i];
    T e2 = edges[i+1];

    index[e1].push_back(e2);
    index[e2].push_back(e1);
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
  
  std::vector<bool> visited(Ne, false);

  while (!stack.empty()) {
    node = stack.top();
    parent = parents.top();
    depth = depth_stack.top();

    stack.pop();
    parents.pop();
    depth_stack.pop();

    while (path.size() && path.size() > depth) {
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

  // find start of loop
  size_t i;
  for (i = 0; i < vec_path.size(); i++) {
    if (vec_path[i] == node) {
      break;
    }
  }

  if (vec_path.size() - i < 3) {
    return std::vector<T>(0);
  }

  const size_t new_len = vec_path.size() - i;

  std::vector<T> elist((new_len - 1) * 2);
  elist.reserve((new_len - 1) * 2);
  for (size_t j = 0; j < new_len - 1; j++) {
    T e1 = vec_path[i + j];
    T e2 = vec_path[i + j + 1];

    if (e1 > e2) {
      elist[2*j] = e1;
      elist[2*j + 1] = e2;      
    }
    else {
      elist[2*j] = e2;
      elist[2*j + 1] = e1;      
    }
  }

  std::reverse(elist.begin(), elist.end());

  return elist;
}

};

#endif