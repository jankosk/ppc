#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

void print_vec(const std::vector<float>& vec, int ny, int nx) {
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            std::cout << vec[y + x * ny] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n";
}

__global__ void preprocess_kernel(float* result, float* data, int ny, int nx, int ny_padded) {
    int y = threadIdx.x + blockIdx.x * 64;

    if (y >= ny) {
        for (int x = 0; x < nx; ++x) {
            result[y + x * ny_padded] = 0;
        }
        return;
    }

    float mean = 0;
    for (int x = 0; x < nx; ++x) {
        mean += data[x + y * nx];
    }
    mean = mean / nx;
    float stdiv = 0;
    for (int x = 0; x < nx; ++x) {
        float diff = data[x + y * nx] - mean;
        stdiv += diff * diff;
    }
    stdiv = sqrt(stdiv);
    for (int x = 0; x < nx; ++x) {
        int idx = y + x * ny_padded;
        result[idx] = (data[x + y * nx] - mean) / stdiv;
    }
}

__global__ void kernel(float* result, float* data, int ny, int nx, int ny_padded) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    float sums[8][8];

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            sums[i][j] = 0;
        }
    }

    for (int x = 0; x < nx; ++x) {
        float col_x[8];
        float col_y[8];
        for (int chunk_i = 0; chunk_i < 8; chunk_i++) {
            int i = bx * 64 + chunk_i * 8 + tx;
            col_x[chunk_i] = data[i + x * ny_padded];
        }
        for (int chunk_j = 0; chunk_j < 8; chunk_j++) {
            int j = by * 64 + chunk_j * 8 + ty;
            col_y[chunk_j] = data[j + x * ny_padded];
        }
        for (int chunk_i = 0; chunk_i < 8; chunk_i++) {
            for (int chunk_j = 0; chunk_j < 8; chunk_j++) {
                sums[chunk_i][chunk_j] += col_x[chunk_i] * col_y[chunk_j];
            }
        }
    }

    for (int chunk_i = 0; chunk_i < 8; chunk_i++) {
        for (int chunk_j = 0; chunk_j < 8; chunk_j++)  {
            int i = bx * 64 + chunk_i * 8 + tx;
            int j = by * 64 + chunk_j * 8 + ty;
            if (i < ny && j < ny) {
                result[i + j * ny] = sums[chunk_i][chunk_j];
            }
        }
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
    int block_chunk = 64;
    int num_block_chunks = divup(ny, block_chunk);
    int ny_padded = roundup(ny, block_chunk);

    float* dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, nx * ny_padded * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    float* preprocessedGPU = NULL;
    CHECK(cudaMalloc((void**)&preprocessedGPU, nx * ny_padded * sizeof(float)));

    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));

    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(num_block_chunks, 1);
        preprocess_kernel<<<dimGrid, dimBlock>>>(preprocessedGPU, dataGPU, ny, nx, ny_padded);
        CHECK(cudaGetLastError());
    }

    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(num_block_chunks, num_block_chunks);
        kernel<<<dimGrid, dimBlock>>>(resultGPU, preprocessedGPU, ny, nx, ny_padded);
        CHECK(cudaGetLastError());
    }


    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(preprocessedGPU));
    CHECK(cudaFree(resultGPU));
}
