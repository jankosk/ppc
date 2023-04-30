#include <iostream>
#include <cmath>
#include <vector>
#include <x86intrin.h>

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

constexpr float8_t init_f8 {
    0, 0, 0, 0,
    0, 0, 0, 0
};

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int col_chunk = 8;
    int num_col_chunks = (ny + col_chunk - (ny % col_chunk)) / col_chunk;
    int ny_padded = num_col_chunks * col_chunk;
    std::vector<float> norm(nx * ny);
    std::vector<float> norm_padded(nx * ny_padded);
    std::vector<float8_t> norm_chunk(nx * num_col_chunks);

    #pragma omp parallel for
    for (int y = 0; y < ny; ++y) {
        float mean = 0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];
        }
        mean = mean / nx;
        float stdiv = 0;
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
            float8_t prod000 = init_f8;
            float8_t prod001 = init_f8;
            float8_t prod010 = init_f8;
            float8_t prod011 = init_f8;
            float8_t prod100 = init_f8;
            float8_t prod101 = init_f8;
            float8_t prod110 = init_f8;
            float8_t prod111 = init_f8;
            for (int x = 0; x < nx; ++x) {
                constexpr int PF = 20;
                __builtin_prefetch(&norm_chunk[nx * chunk_i + x + PF]);
                __builtin_prefetch(&norm_chunk[nx * chunk_j + x + PF]);
                float8_t a000 = norm_chunk[nx * chunk_i + x];
                float8_t b000 = norm_chunk[nx * chunk_j + x];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);
                prod000 += a000 * b000;
                prod001 += a000 * b001;
                prod010 += a010 * b000;
                prod011 += a010 * b001;
                prod100 += a100 * b000;
                prod101 += a100 * b001;
                prod110 += a110 * b000;
                prod111 += a110 * b001;
            }
            float8_t summed[col_chunk] = {
                prod000, swap1(prod001), prod010, swap1(prod011),
                prod100, swap1(prod101), prod110, swap1(prod111)
            };
            for (int ii = 0; ii < col_chunk; ++ii) {
                for (int jj = 0; jj < col_chunk; ++jj) {
                    int i = ii + chunk_i * col_chunk;
                    int j = jj + chunk_j * col_chunk;
                    if (i < ny && j < ny) {
                        result[j + i * ny] = summed[jj^ii][jj];
                    }
                }
            }
        }
    }
}