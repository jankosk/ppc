#include <iostream>
#include <algorithm>
#include <vector>

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
	#pragma omp parallel for
    for (int y = 0; y < ny; ++y) {
    	std::vector<float> window;

        for (int x = 0; x < nx; ++x) {
            window.clear();
            int a_min = std::max(x - hx, 0);
            int a_max = std::min(x + hx + 1, nx);
            int b_min = std::max(y - hy, 0);
            int b_max = std::min(y + hy + 1, ny);

            for (int i = b_min; i < b_max; ++i) {
                for (int j = a_min; j < a_max; ++j) {
                    window.push_back(in[j + i * nx]);
                }
            }
            int size = window.size();
            int size_2 = size / 2;
            std::nth_element(window.begin(), window.begin() + size_2, window.end());

            if (size % 2 == 0) {
                float median1 = window[size_2];
                std::nth_element(window.begin(), window.begin() + size_2 - 1, window.end());
                float median2 = window[size_2 - 1];
                out[x + y * nx] = (median1 + median2) / 2;
            } else {
                out[x + y * nx] = window[size_2];
            }
        }
    }
}