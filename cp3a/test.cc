#include <random>
#include "cp.hh"


int main() {
    int ny = 5;
    int nx = 2;
    float data[nx * ny];
    float result[nx * ny];
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            data[y + x * ny] = rand() / static_cast<float>(RAND_MAX);
        }
    }
    correlate(ny, nx, data, result);

    return 0;
}