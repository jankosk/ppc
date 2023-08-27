#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

constexpr int parallel_sort_threshold = 1000;

data_t median3(data_t a, data_t b, data_t c) {
    if ((a > b) ^ (a > c)) return a;
    if ((b < a) ^ (b < c)) return b;
    return c;
}

void quick_sort(int n, data_t *data) {
    if (n < 2) return;
    if (n < parallel_sort_threshold) {
        std::sort(data, data+n);
        return;
    }
    auto pivot = median3(data[0], data[n / 2], data[n - 1]);
    auto *left = data;
    auto *right = data + n - 1;

    while (left <= right) {
        while (*left < pivot) left++;
        while (*right > pivot) right--;
        if (left <= right) {
            std::swap(*left, *right);
            left++;
            right--;
        }
    }
    #pragma omp task
    quick_sort(right - data + 1, data);
    #pragma omp task
    quick_sort(n - (left - data), left);
}

void psort(int n, data_t *data) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            quick_sort(n, data);
        }
    }
}