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

__global__ void kernel(float* result, float* data, int ny, int nx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny) {
        return;
    }
    float sum = 0;
    for (int x = 0; x < nx; ++x) {
        sum += data[x + i * nx] * data[x + j * nx];
    }
    result[i + j * ny] = sum;
}

void preprocess(std::vector<float>& normalized, const float* data, int ny, int nx) {
    for (int y = 0; y < ny; ++y) {
        float mean = 0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];
        }
        mean = mean / nx;
        float stdiv = 0;
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
    std::vector<float> normalized_data(ny * nx);
    preprocess(normalized_data, data, ny, nx);
    float* normalized_data_ptr = normalized_data.data();

    float* dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, nx * ny * sizeof(float)));
    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, normalized_data_ptr, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    kernel<<<dimGrid, dimBlock>>>(resultGPU, dataGPU, ny, nx);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(resultGPU));
}
