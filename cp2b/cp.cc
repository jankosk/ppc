#include <iostream>
#include <cmath>
#include <vector>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int chunk = 12;
    int num_chunks = (nx + chunk - (nx % chunk)) / chunk;
    int nx_padded = chunk * num_chunks;
    std::vector<double> normalized(ny * nx);
    std::vector<double> normalized_padded(ny * nx_padded);

    #pragma omp parallel for
    for (int y = 0; y < ny; ++y) {
        double mean = 0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];   
        }
        mean = mean / nx;
        double stdiv = 0;
        for (int x = 0; x < nx; ++x) {
            int idx = x + y * nx;
            normalized[idx] = data[idx] - mean;
            stdiv += pow(data[idx] - mean, 2);
        }
        stdiv = sqrt(stdiv);
        for (int x = 0; x < nx; ++x) {
            int idx = x + y * nx;
            normalized[idx] = normalized[idx] / stdiv;
        }
    }

    #pragma omp parallel for
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx_padded; ++x) {
            if (x < nx) {
                normalized_padded[x + y * nx_padded] = normalized[x + y * nx];
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        for (int j = i; j < ny; ++j) {
            double sums[chunk]{0};
            for (int ka = 0; ka < num_chunks; ++ka) {
                for (int kb = 0; kb < chunk; ++kb) {
                    double a = normalized_padded[kb + ka * chunk + i * nx_padded];
                    double b = normalized_padded[kb + ka * chunk + j * nx_padded];
                    sums[kb] += a * b;
                }
            }
            double sum = 0;
            for (int kb = 0; kb < chunk; ++kb) {
                sum += sums[kb];
            }
            result[i + j * ny] = sum;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (j < i) {
                result[i + j * ny] = result[j + i * ny];
            }
        }
    }
}
