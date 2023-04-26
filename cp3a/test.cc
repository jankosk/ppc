#include <random>
#include "cp.hh"


int main() {
    constexpr int ny = 5;
    constexpr int nx = 2;
    float data[ny * nx] = {
        0.81472367, 0.90579194,
        0.45150527, 0.49610928,
        0.96488851, 0.15761308,
        1.52930284, 0.26519677,
        2.92820811, 1.35169840,
    };
    float result[nx * ny];

    correlate(ny, nx, data, result);

    return 0;
}