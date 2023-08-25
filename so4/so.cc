#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

constexpr int parallel_sort_threshold = 10000;

void merge_sort(int n, data_t *data) {
    if(n < parallel_sort_threshold) {
        std::sort(data, data + n);
        return;
    }
    int middle = (n + 1) / 2;
    #pragma omp task
    merge_sort(middle, data);
    #pragma omp task
    merge_sort(n - middle, data + middle);
    #pragma omp taskwait
    std::inplace_merge(data, data + middle, data + n);
}

void psort(int n, data_t *data) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            merge_sort(n, data);
        }
    }
}