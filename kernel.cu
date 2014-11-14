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
#include "support.h"
#include<iostream>
using namespace std;
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
    block_dim.x = BLOCK_SIZE; 
    block_dim.y = 1; block_dim.z = 1;
    grid_dim.x = ceil(num_items_in_transactions / (16.0 * BLOCK_SIZE)); 
    grid_dim.y = 1; grid_dim.z = 1;
    if (max_unique_items * sizeof(unsigned int) < SM_PER_BLOCK) {
        // private histogram should fit in shared memory
        histogram_kernel<<<grid_dim, block_dim, max_unique_items * sizeof(unsigned int)>>>(d_transactions, d_flist, num_items_in_transactions);
    } else {
        // private histogram will not fit in shared memory. launch global kernel
        histogram_kernel_naive<<<grid_dim, block_dim>>>(d_transactions, d_flist, num_items_in_transactions, max_unique_items);
    }
    
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
}
    
   
   
   
__global__ void sort_transaction_kernel(unsigned short *d_flist_key_16_index, unsigned int *d_flist, unsigned int *d_transactions,
        unsigned int *offset_array, unsigned int num_transactions, unsigned int num_elements, unsigned int bins, bool indexFileInConstantMem) {
   
    //unsigned int transaction_index = threadIdx.x + blockDim.x * blockIdx.x;
    //unsigned int stride = blockDim.x * gridDim.x;
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int swap = 0;
    unsigned int start_offset = 0;
    unsigned int end_offset = 0;
    unsigned int index1 = 0;
    unsigned int transaction_start_index = blockDim.x * blockIdx.x;
    //TBD: need to pass dynamically
    __shared__ unsigned int Ts[TRANSACTION_PER_SM][max_items_in_transaction];
    
    while (transaction_start_index < num_transactions) {
    unsigned int index = threadIdx.x;
    unsigned int transaction_end_index = transaction_start_index +  blockDim.x;
    
    __syncthreads();
    // clear SM 
    for (i = 0; i < TRANSACTION_PER_SM; i++) {
        while (index < max_items_in_transaction) {
            Ts[i][index] = 0;//INVALID;
            index += blockDim.x;
        }
        __syncthreads();
    }
    // get all the transaction assigned to this block into SM
    for (i = transaction_start_index; i < transaction_end_index && i < num_transactions; i++) {
        // get the ith transaction data into SM
        start_offset = offset_array[i];
        end_offset = offset_array[i+1];
        index1 = start_offset + threadIdx.x;
        __syncthreads();
        // threads collaborate to get the ith transaction
        while (index1 < end_offset) {
            Ts[i-transaction_start_index][index1 - start_offset] = d_transactions[index1];        
            index1 += blockDim.x;
        }
        __syncthreads();
    }

    // now that all transactions are in SM, each thread takes ownership of a row of SM
    // (i.e. one transaction per thread)
    if (threadIdx.x < TRANSACTION_PER_SM) {
        //to test basic functionality
        /*for (int i =0; i < max_items_in_transaction;i++) {
            if (Ts[threadIdx.x][i] < INVALID) {
                Ts[threadIdx.x][i]++;
            }
        }*/
        for (i= 0; i < (max_items_in_transaction - 1);i++) {
            for (j = 0;j < (max_items_in_transaction - 1 - i);j++) {
                //if (Ts[threadIdx.x][j] == INVALID || Ts[threadIdx.x][j + 1] == INVALID) {
                    // all subsequent element will be invalid
                 //   goto endloop;
               // }
                if (dc_flist_key_16_index[Ts[threadIdx.x][j]] > dc_flist_key_16_index[Ts[threadIdx.x][j + 1]]) {
                    // this means index of jth element is greater than index of j+1th element in index array,
                    // which implies jth element has a lesser count than j+1th element.
                    // so swap them.
                    // this also ensures automatically pruned items (having index 0xFFFF) will get pushed back.
                    // while making fp tree we can choose to ignore all items which have index 0xFFFF
                    swap = Ts[threadIdx.x][j];
                    Ts[threadIdx.x][j] = Ts[threadIdx.x][j + 1];
                    Ts[threadIdx.x][j + 1] = swap;
                }
            } 
        }
    }
//endloop:
    __syncthreads();
    // now that work is done write back results 
    for (i = transaction_start_index; i < transaction_end_index && i < num_transactions; i++) {
        // get the ith transaction data from SM to global mem
        start_offset = offset_array[i];
        end_offset = offset_array[i+1];
        index1 = start_offset + threadIdx.x;
        __syncthreads();
        while (index1 < end_offset) {
            d_transactions[index1] = Ts[i - transaction_start_index][index1 - start_offset];        
            index1 += blockDim.x;
        }
        __syncthreads();
    }
    transaction_start_index += (blockDim.x * gridDim.x);
    }
} 

void sort_transaction(unsigned short *d_flist_key_16_index, unsigned int *d_flist, unsigned int *d_transactions, unsigned int *offset_array, unsigned int num_transactions, unsigned int num_items_in_transactions, unsigned int bins,bool indexFileInConstantMem) {
    cudaDeviceProp deviceProp;
    cudaError_t ret;
    cudaGetDeviceProperties(&deviceProp, 0);
    int SM_PER_BLOCK = deviceProp.sharedMemPerBlock;
    
    dim3 block_dim;
    dim3 grid_dim;
    
    unsigned int bytesPerTransaction = max_items_in_transaction * sizeof(unsigned int);
    
    block_dim.x = ((SM_PER_BLOCK / bytesPerTransaction) - 10) > TRANSACTION_PER_SM ? TRANSACTION_PER_SM : ((SM_PER_BLOCK / bytesPerTransaction) - 10);
    block_dim.y = 1;
    block_dim.y = 1;

    grid_dim.x = (int) ceil(num_transactions / (2.0 * block_dim.x));
    grid_dim.y = 1;
    grid_dim.z = 1;
#ifdef TEST_MODE
    cout<<"sort_transaction_kernel<bx,gx>"<<block_dim.x<<","<<grid_dim.x<<endl;
#endif
    sort_transaction_kernel<<<grid_dim, block_dim>>>(d_flist_key_16_index, d_flist, d_transactions, offset_array,
            num_transactions, num_items_in_transactions, bins, indexFileInConstantMem); 
    ret = cudaDeviceSynchronize();
    if(ret != cudaSuccess) FATAL("Unable to launch kernel");
    
    
}  
