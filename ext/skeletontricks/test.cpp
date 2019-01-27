#include "skeletontricks.hpp"
#include <cstdint>

int main() {

  int voxels = 512*512*512;
  uint8_t *labels = new uint8_t[voxels];
  float *DBF = new float[voxels];

  for (int i = 0; i < voxels; i++) {
    labels[i] = 1;
    DBF[i] = 1;
  }

  int path[50];
  for (int i = 0; i < 50; i++) {
    path[i] = i * 100000;
  }

  for (int i = 0; i < 50; i++) {
    skeletontricks::_roll_invalidation_cube(
      labels, DBF,
      512, 512, 512,
      1.0, 1.0, 1.0,
      path, 50,
      10, 40
    );
  }

  return 0;
}