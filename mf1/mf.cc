#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>

void print_vec(const std::vector<float>& vec, int ny, int nx) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            std::cout << vec[x + y * nx] << " ";
        }
        std::cout << std::endl;
    }
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int a_min = std::max(x - hx, 0);
            int a_max = std::min(x + hx + 1, nx);
            int b_min = std::max(y - hy, 0);
            int b_max = std::min(y + hy + 1, ny);
            int a_diff = a_max - a_min;
            int b_diff = b_max - b_min;

            int size = a_diff * b_diff;
            std::vector<float> vals(size);
            int k = 0;
            for (int i = b_min; i < b_max; ++i) {
                for (int j = a_min; j < a_max; ++j) {
                    vals[k] = in[j + i * nx];
                    ++k;
                }
            }
            std::sort(vals.begin(), vals.end());

            if (size % 2 == 0) {
                int idx = (size - 1) / 2;
                float median1 = vals[idx];
                float median2 = vals[idx + 1];
                out[x + y * nx] = (median1 + median2) / 2;
            } else {
                int idx = (size - 1) / 2;
                out[x + y * nx] = vals[idx];
            }
        }
    }
}