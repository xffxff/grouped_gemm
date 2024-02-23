/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"

template <typename T>
__global__ void moe_permute_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *map_dest_row_to_source_row,
                                   int *map_source_row_to_dest_row,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = map_dest_row_to_source_row[dest_row];

    if (threadIdx.x == 0)
    {
        // write the map for the following unpermuting
        map_source_row_to_dest_row[source_row] = dest_row;
    }

    // permute activations rows based on experts
    const int kElementsPerAccess = 16 / sizeof(T);
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x * kElementsPerAccess; tid < num_cols; tid += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<float4, sizeof(float4), cutlass::arch::CacheOperation::LastUse>(
            *(float4 *)(dest_row_ptr + tid), (source_row_ptr + tid), true);
    }
}

template <typename T>
__global__ void moe_recover_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *map_dest_row_to_source_row,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = map_dest_row_to_source_row[dest_row];

    // permute activations rows based on experts
    const int kElementsPerAccess = 16 / sizeof(T);
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x * kElementsPerAccess; tid < num_cols; tid += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<float4, sizeof(float4), cutlass::arch::CacheOperation::LastUse>(
            *(float4 *)(dest_row_ptr + tid), (source_row_ptr + tid), true);
    }
}

template <typename T>
void moe_permute_kernel_launcher(
    const T *original_input,
    T *permuted_output,
    const int *map_dest_row_to_source_row,
    int *map_source_row_to_dest_row,
    const int num_rows,
    const int num_cols,
    cudaStream_t stream)
{
    const int blocks = num_rows;
    const int threads = std::min(num_cols, 1024);

    if (map_source_row_to_dest_row != nullptr)
    {
        moe_permute_kernel<T><<<blocks, threads, 0, stream>>>(original_input,
                                                              permuted_output,
                                                              map_dest_row_to_source_row,
                                                              map_source_row_to_dest_row,
                                                              num_rows,
                                                              num_cols);
    }
    else
    {
        moe_recover_kernel<T><<<blocks, threads, 0, stream>>>(original_input,
                                                              permuted_output,
                                                              map_dest_row_to_source_row,
                                                              num_rows,
                                                              num_cols);
    }
}