#include <algorithm>
#include <vector>
#include <iostream>
#include <omp.h>

typedef unsigned long long data_t;

void print(const data_t* arr, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
}

void merge(data_t* arr, data_t* temp, int left_idx, int middle_idx, int right_idx) {
    int size_left = middle_idx - left_idx + 1;
    int size_right = right_idx - middle_idx;

    for (int i = 0; i < size_left; ++i) {
        temp[left_idx + i] = arr[left_idx + i];
    }
    for (int i = 0; i < size_right; ++i) {
        temp[middle_idx + 1 + i] = arr[middle_idx + 1 + i];
    }

    int i = left_idx;
    int j = middle_idx + 1;
    int k = left_idx;
    while (i <= middle_idx && j <= right_idx) {
        if (temp[i] < temp[j]) {
            arr[k] = temp[i];
            ++i;
        } else {
            arr[k] = temp[j];
            ++j;
        }
        ++k;
    }
    while (i <= middle_idx) {
        arr[k] = temp[i];
        ++i;
        ++k;
    }
    while (j <= right_idx) {
        arr[k] = temp[j];
        ++j;
        ++k;
    }
}

void merge_sort(data_t* arr, data_t* temp, int left_idx, int right_idx) {
    if (left_idx >= right_idx) {
        return;
    }
    int middle_idx = left_idx + (right_idx - left_idx) / 2;
    merge_sort(arr, temp, left_idx, middle_idx);
    merge_sort(arr, temp, middle_idx + 1, right_idx);
    merge(arr, temp, left_idx, middle_idx, right_idx);
}

void psort(int n, data_t *data) {
    int max_threads = omp_get_max_threads();
    int threads = n < max_threads * 2 ? 1 : max_threads;
    int chunk = (n + threads - 1) / threads;
    data_t* temp = new data_t[n];
    #pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        int begin = chunk * i;
        int end = std::min(begin + chunk - 1, n - 1);
        merge_sort(data, temp, begin, end);
    }
    for (int i = 1; i < threads; ++i) {
        int end = std::min(i * chunk + chunk - 1, n - 1);
        int middle = std::min(i * chunk - 1, n - 1);
        merge(data, temp, 0, middle, end);
    }
    delete[] temp;
}
