#include <iostream>
#include <cmath>
#include <vector>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

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
    constexpr int row_chunk = 4;
    int num_row_chunks = (nx + row_chunk - (nx % row_chunk)) / row_chunk;
    int nx_padded = num_row_chunks * row_chunk;
    constexpr int col_chunk = 6;
    int num_col_chunks = (ny + col_chunk - (ny % col_chunk)) / col_chunk;
    int ny_padded = num_col_chunks * col_chunk;
    std::vector<double> norm(nx * ny);
    std::vector<double> norm_padded(nx_padded * ny_padded);
    std::vector<double4_t> norm_d4(num_row_chunks * ny_padded);

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
        for (int x = 0; x < nx_padded; ++x) {
            bool is_value = x < nx && y < ny;
            norm_padded[x + y * nx_padded] = is_value ? norm[x + y * nx] : 0;
        }
    }

    #pragma omp parallel for
    for (int y = 0; y < ny_padded; ++y) {
        for (int ka = 0; ka < num_row_chunks; ++ka) {
            for (int kb = 0; kb < row_chunk; ++kb) {
                int x = kb + ka * row_chunk;
                norm_d4[num_row_chunks * y + ka][kb] = norm_padded[x + y * nx_padded];
            }
        }
    }

    #pragma omp parallel for
    for (int ic = 0; ic < num_col_chunks; ++ic) {
        for (int jc = 0; jc < num_col_chunks; ++jc) {
            double4_t products[col_chunk][col_chunk];
            for (int id = 0; id < col_chunk; ++id) {
                for (int jd = 0; jd < col_chunk; ++jd) {
                    products[id][jd] = init_double4;
                }
            }

            for (int ka = 0; ka < num_row_chunks; ++ka) {
                double4_t y0 = norm_d4[num_row_chunks * (jc * col_chunk + 0) + ka];
                double4_t y1 = norm_d4[num_row_chunks * (jc * col_chunk + 1) + ka];
                double4_t y2 = norm_d4[num_row_chunks * (jc * col_chunk + 2) + ka];
                double4_t y3 = norm_d4[num_row_chunks * (jc * col_chunk + 3) + ka];
                double4_t y4 = norm_d4[num_row_chunks * (jc * col_chunk + 4) + ka];
                double4_t y5 = norm_d4[num_row_chunks * (jc * col_chunk + 5) + ka];
                double4_t x0 = norm_d4[num_row_chunks * (ic * col_chunk + 0) + ka];
                double4_t x1 = norm_d4[num_row_chunks * (ic * col_chunk + 1) + ka];
                double4_t x2 = norm_d4[num_row_chunks * (ic * col_chunk + 2) + ka];
                double4_t x3 = norm_d4[num_row_chunks * (ic * col_chunk + 3) + ka];
                double4_t x4 = norm_d4[num_row_chunks * (ic * col_chunk + 4) + ka];
                double4_t x5 = norm_d4[num_row_chunks * (ic * col_chunk + 5) + ka];
                products[0][0] += x0 * y0;
                products[0][1] += x0 * y1;
                products[0][2] += x0 * y2;
                products[0][3] += x0 * y3;
                products[0][4] += x0 * y4;
                products[0][5] += x0 * y5;
                products[1][0] += x1 * y0;
                products[1][1] += x1 * y1;
                products[1][2] += x1 * y2;
                products[1][3] += x1 * y3;
                products[1][4] += x1 * y4;
                products[1][5] += x1 * y5;
                products[2][0] += x2 * y0;
                products[2][1] += x2 * y1;
                products[2][2] += x2 * y2;
                products[2][3] += x2 * y3;
                products[2][4] += x2 * y4;
                products[2][5] += x2 * y5;
                products[3][0] += x3 * y0;
                products[3][1] += x3 * y1;
                products[3][2] += x3 * y2;
                products[3][3] += x3 * y3;
                products[3][4] += x3 * y4;
                products[3][5] += x3 * y5;
                products[4][0] += x4 * y0;
                products[4][1] += x4 * y1;
                products[4][2] += x4 * y2;
                products[4][3] += x4 * y3;
                products[4][4] += x4 * y4;
                products[4][5] += x4 * y5;
                products[5][0] += x5 * y0;
                products[5][1] += x5 * y1;
                products[5][2] += x5 * y2;
                products[5][3] += x5 * y3;
                products[5][4] += x5 * y4;
                products[5][5] += x5 * y5;
            }

            for (int id = 0; id < col_chunk; ++id) {
                for (int jd = 0; jd < col_chunk; ++jd) {
                    int i = ic * col_chunk + id;
                    int j = jc * col_chunk + jd;
                    if (i < ny && j < ny) {
                        result[ny * i + j] = sum_double4(products[id][jd]);
                    }
                }
            }
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