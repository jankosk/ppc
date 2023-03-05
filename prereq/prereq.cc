#include <iostream>

struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    Result result{{0.0f, 0.0f, 0.0f}};
    const int area = (x1 - x0) * (y1 - y0);
    double reds = 0;
    double greens = 0;
    double blues = 0;

    for (int x = x0; x < x1; x++) {
        for (int y = y0; y < y1; y++) {
          reds += data[0 + 3 * x + 3 * nx * y];
          greens += data[1 + 3 * x + 3 * nx * y];
          blues += data[2 + 3 * x + 3 * nx * y];
        }
    }
    reds = reds / area;
    greens = greens / area;
    blues = blues / area;

    result = {(float) reds, (float) greens, (float) blues};
    return result;
}
