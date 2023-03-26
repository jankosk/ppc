#include <iostream>
#include <cmath>
#include <vector>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

void print_double4(const double4_t& d) {
    for (int i = 0; i < 4; ++i) {
        std::cout << " " << d[i] << " ";
    } 
}

void print_vec(const std::vector<double4_t>& vec, int na, int nb) {
    int i = 0;
    for (const auto &d : vec) {
        if (i % (na * nb) == 0) {
            std::cout << std::endl;
        }
        i += nb;
        print_double4(d);
    }
    std::cout << std::endl;
}

void print_vec(const std::vector<double>& vec, int ny, int nx) {
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
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int nb = 4;
    int na = (nx + nb - (nx % nb)) / nb;
    int nab = na * nb;
    std::vector<double> norm(nx * ny);
    std::vector<double> norm_padded(nab * ny);
    std::vector<double4_t> norm_d4(na * ny);

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

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nab; ++x) {
            norm_padded[x + y * nab] = x < nx ? norm[x + y * nx] : 0;
        }
    }

    for (int y = 0; y < ny; ++y) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int x = kb + ka * nb;
                norm_d4[na * y + ka][kb] = norm_padded[x + y * nab];
            }
        }
    }

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (j < i) {
                result[i + j * ny] = result[j + i * ny];
                continue;
            }
            double4_t sums = {0,0,0,0};
            asm("# loop starts here");
            for (int ka = 0; ka < na; ++ka) {
                double4_t a = norm_d4[na * i + ka];
                double4_t b = norm_d4[na * j + ka];
                sums += a * b;
            }
            asm("# loop ends here");
            double sum = 0;
            for (int kb = 0; kb < nb; ++kb) {
                sum += sums[kb];
            }
            result[i + j * ny] = sum;
        }
    }
}