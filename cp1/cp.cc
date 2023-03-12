#include <algorithm>
#include <iostream>
#include <cmath>

void print_array(double *arr, int ny, int nx) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            std::cout << arr[y * nx + x] << " ";
        }
        std::cout << std::endl;
    }
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    double *normalized = new double[ny * nx];
    for (int y = 0; y < ny; ++y) {
        double mean = 0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];
        }
        mean = mean / nx;
        double stdiv = 0;
        for (int x = 0; x < nx; ++x) {
            normalized[x + y * nx] = data[x + y * nx] - mean;
            stdiv += pow(data[x + y * nx] - mean, 2);
        }
        stdiv = sqrt(stdiv);
        for (int x = 0; x < nx; ++x) {
            int idx = x + y * nx;
            normalized[idx] = normalized[idx] / stdiv;
        }
    }
    // print_array(normalized, ny, nx);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (j < i) {
                result[i + j * ny] = result[j + i * ny];
                continue;
            }
            double sum = 0;
            for (int x = 0; x < nx; ++x) {
                sum += normalized[x + i * nx] * normalized[x + j * nx];
            }
            result[i + j * ny] = sum;
        }
    }
    delete[] normalized;
}
