/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE


#include "defs.h"
#include "kernel_radix.cu"
__constant__ unsigned short dc_flist_key_16_index[max_unique_items];
__global__ void histogram_kernel_naive(unsigned int* input, unsigned int* bins,
        unsigned int num_elements, unsigned int num_bins) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (i < num_elements) {
        int bin_num = input[i];
        if (bin_num < num_bins) {
            atomicAdd(&bins[bin_num], 1);
        }
        i+=stride;
    }
}
__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
        unsigned int num_elements) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int index_x = 0;
    extern __shared__ unsigned int hist_priv[];
    for (int i = 0; i < ceil(max_unique_items / (1.0 * blockDim.x)); i++){
        index_x = threadIdx.x + i * blockDim.x;
        if (index_x < max_unique_items )
            hist_priv[index_x] = 0;
    }

    __syncthreads();
    unsigned int stride = blockDim.x * gridDim.x;
    while (i < num_elements) {
        int bin_num = input[i];
        if (bin_num < max_unique_items ) {
            atomicAdd(&hist_priv[bin_num], 1);
        }
        i+=stride;
    }
    __syncthreads();
    for (int i = 0; i < ceil(max_unique_items / (1.0 * blockDim.x)); i++){
        index_x = threadIdx.x + i * blockDim.x;
        if (index_x < max_unique_items) {
            atomicAdd(&bins[index_x], hist_priv[index_x]);
        }
    }
}

    //make_flist(d_trans_offsets, d_transactions, d_flist, num_transactions, num_items_in_transactions);
void make_flist(unsigned int *d_trans_offset, unsigned int *d_transactions, unsigned int *d_flist,
        unsigned int num_transactions, unsigned int num_items_in_transactions, int SM_PER_BLOCK) {
    
    cudaError_t cuda_ret;
    dim3 grid_dim, block_dim;
/*
    unsigned int *temp_out_scan_d;
    unsigned int *temp_out_d;


    cuda_ret = cudaMalloc((void**)&temp_out_scan_d, max_unique_items * sizeof(unsigned int ));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate scan memory");
    
    cuda_ret = cudaMalloc((void**)&temp_out_d, max_unique_items * sizeof(unsigned int ));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate scan memory");
    
    cuda_ret = cudaMemset(temp_out_scan_d, 0, max_unique_items * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
*/
    block_dim.x = BLOCK_SIZE; 
    block_dim.y = 1; block_dim.z = 1;
    grid_dim.x = ceil(num_items_in_transactions / (16.0 * BLOCK_SIZE)); 
    grid_dim.y = 1; grid_dim.z = 1;
    //printf("<bx,gx>=%d,%d\n", block_dim.x, grid_dim.x);
    if (max_unique_items * sizeof(unsigned int) < SM_PER_BLOCK) {
        // private histogram should fit in shared memory
        histogram_kernel<<<grid_dim, block_dim, max_unique_items * sizeof(unsigned int)>>>(d_transactions, d_flist, num_items_in_transactions);
    } else {
        // private histogram will not fit in shared memory. launch global kernel
        histogram_kernel_naive<<<grid_dim, block_dim>>>(d_transactions, d_flist, num_items_in_transactions, max_unique_items);
    }
    
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    //printf("kernel launch success");
    
    // radix sort the items list 
    //radix_sort(d_flist, temp_out_d, temp_out_scan_d, max_unique_items);
}
