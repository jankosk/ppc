#include <iostream>
#include <cmath>
#include <vector>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_t_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

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
    constexpr int nb = 4;
    int na = (nx + nb - (nx % nb)) / nb;
    int nab = na * nb;
    constexpr int nd = 3;
    int nc = (ny + nd - (ny % nd)) / nd;
    int ncd = nc * nd;
    std::vector<double> norm(nx * ny);
    std::vector<double> norm_padded(nab * ncd);
    double4_t* norm_d4 = double4_t_alloc(na * ncd);

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
    for (int y = 0; y < ncd; ++y) {
        for (int x = 0; x < nab; ++x) {
            bool is_value = x < nx && y < ny;
            norm_padded[x + y * nab] = is_value ? norm[x + y * nx] : 0;
        }
    }

    #pragma omp parallel for
    for (int y = 0; y < ncd; ++y) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int x = kb + ka * nb;
                norm_d4[na * y + ka][kb] = norm_padded[x + y * nab];
            }
        }
    }

    #pragma omp parallel for
    for (int ic = 0; ic < nc; ++ic) {
        for (int jc = 0; jc < nc; ++jc) {
            double4_t products[nd][nd];
            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    products[id][jd] = init_double4;
                }
            }

            for (int ka = 0; ka < na; ++ka) {
                double4_t y0 = norm_d4[na * (jc * nd + 0) + ka];
                double4_t y1 = norm_d4[na * (jc * nd + 1) + ka];
                double4_t y2 = norm_d4[na * (jc * nd + 2) + ka];
                double4_t x0 = norm_d4[na * (ic * nd + 0) + ka];
                double4_t x1 = norm_d4[na * (ic * nd + 1) + ka];
                double4_t x2 = norm_d4[na * (ic * nd + 2) + ka];
                products[0][0] += x0 * y0;
                products[0][1] += x0 * y1;
                products[0][2] += x0 * y2;
                products[1][0] += x1 * y0;
                products[1][1] += x1 * y1;
                products[1][2] += x1 * y2;
                products[2][0] += x2 * y0;
                products[2][1] += x2 * y1;
                products[2][2] += x2 * y2;
            }

            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    int i = ic * nd + id;
                    int j = jc * nd + jd;
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

    std::free(norm_d4);
}