#include <iostream>
#include <cmath>
#include <vector>
#include <x86intrin.h>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static inline double4_t swap2(double4_t x) { return _mm256_permute_pd(x, 0b0100); }
static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b1011); }

constexpr double4_t init_double4 {0, 0, 0, 0};

double sum_double4(double4_t d) {
    double sum = 0;
    for (int i = 0; i < 4; ++i) {
        sum += d[i];
    }
    return sum;
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
    constexpr int col_chunk = 4;
    int num_col_chunks = (ny + col_chunk - (ny % col_chunk)) / col_chunk;
    int ny_padded = num_col_chunks * col_chunk;
    std::vector<double> norm(nx * ny);
    std::vector<double> norm_padded(nx * ny_padded);
    std::vector<double4_t> norm_chunk(nx * num_col_chunks);

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
            norm[idx] = data[idx] - mean;
            stdiv += pow(data[idx] - mean, 2);
        }
        stdiv = sqrt(stdiv);
        for (int x = 0; x < nx; ++x) {
            int idx = x + y * nx;
            norm[idx] = norm[idx] / stdiv;
        }
    }

    #pragma omp parallel for
    for (int y = 0; y < ny_padded; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (x < nx && y < ny) {
                norm_padded[x + y * nx] = norm[x + y * nx];
            }
        }
    }

    #pragma omp parallel for
    for (int chunk_i = 0; chunk_i < num_col_chunks; ++chunk_i) {
        for (int x = 0; x < nx; ++x) {
            for (int i = 0; i < col_chunk; ++i) {
                int y = chunk_i * col_chunk + i;
                norm_chunk[x + chunk_i * nx][i] = norm_padded[x + y * nx];
            }
        }
    }

    #pragma omp parallel for
    for (int chunk_i = 0; chunk_i < num_col_chunks; ++chunk_i) {
        for (int chunk_j = chunk_i; chunk_j < num_col_chunks; ++chunk_j) { 
            double4_t prod000 = init_double4;
            double4_t prod001 = init_double4;
            double4_t prod010 = init_double4;
            double4_t prod011 = init_double4;
            for (int x = 0; x < nx; ++x) {
                double4_t a000 = norm_chunk[nx * chunk_i + x];
                double4_t b000 = norm_chunk[nx * chunk_j + x];
                double4_t a001 = swap1(a000);
                double4_t a010 = swap2(a000);
                double4_t b001 = swap1(b000);
                prod000 += a000 * b000;
                prod001 += a001 * b000;
                prod010 += a010 * b000;
                prod011 += a010 * b001;
            }
            double4_t summed[col_chunk] = {prod000, prod001, prod010, swap1(prod011)};
            for (int ii = 0; ii < col_chunk; ++ii) {
                for (int jj = 0; jj < col_chunk; ++jj) {
                    int i = ii + chunk_i * col_chunk;
                    int j = jj + chunk_j * col_chunk;
                    if (i < ny && j < ny) {
                        result[j + i * ny] = summed[ii^jj][ii];
                    }
                }
            }
        }
    }
}