#include "defs.h"
#include "kernel_prescan.cu"
__global__ void splitGPU(unsigned int*in_d, unsigned int *out_d, unsigned int in_size, int bit_shift) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    int bit = 0;
    if (index < in_size) {
        bit = in_d[index] & (1 << bit_shift);
        bit = (bit > 0) ? 1 : 0;
        out_d[index] = 1 - bit;
    }

}
__global__ void indexDefine(unsigned int *in_d, unsigned int *rev_bit_d, unsigned int in_size, unsigned int last_input) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    int total_falses = in_d[in_size - 1] + last_input;
    __syncthreads();
    if (index < in_size) {
        if (rev_bit_d[index] == 0) {
            int val = in_d[index];
            in_d[index] = index + 1 - val + total_falses;
        }
    }

}

__global__ void scatterElements(unsigned int *in_d, unsigned int *index_d, unsigned int *out_d, unsigned int in_size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < in_size) {
        unsigned int val = index_d[index];
        if (val < in_size) {
            out_d[val] = in_d[index];
        }
    }

}
/*
__global__ void radix(unsigned int *in, unsigned int *out, int num_elements) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x;
    __shared__ unsigned int priv_ip[max_num_of_transaction];
    unsigned int priv_thread_data[4];// in registers per thread
    unsigned int priv_thread_scan_data[4];// in registers per thread
    unsigned int count = 0;//per thread
    int bit = 0;
    while (index < num_elements) {
          priv_ip[index] = in[index];
          index += stride;
    }
    int last_input = priv_ip[num_elements - 1];
    __syncthreads();
    for (int bit_shift = 0; bit_shift < 1; bit_shift++) {
        //split bit
        count = 0;
        while (index < num_elements) {
            bit = priv_ip[index] & (1 << bit_shift);
            bit = (bit > 0) ? 1 : 0;
            priv_ip[index] = 1 - bit;// flip bits
            priv_thread_data[count++] = priv_ip[index];
            index += stride;
        }
        __syncthreads();
        // preform prescan here
        // scan output to be in priv_ip
        __syncthreads();
        count = 0;
        int total_falses = last_input + priv_ip[num_elements - 1];
        while (index < num_elements) {
            priv_ip[index] = index - priv_ip[index] + total_falses;
            //priv_ip[index] = (priv_thread_data[count++] == 1) ? priv_ip[index] : priv_thread_scan_data[index]; 
            index += stride;
        }
        __syncthreads();
        
    }
}
*/
void radix_sort(unsigned int *in_d, unsigned int *out_d, unsigned int *out_scan_d, int num_elements) {
    cudaError_t ret;
    unsigned int *temp;
    dim3 dimThreadBlock;
    dimThreadBlock.x = BLOCK_SIZE;
    dimThreadBlock.y = 1;
    dimThreadBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x =(int)(ceil(num_elements/(4.0 * dimThreadBlock.x)));
    dimGrid.y = 1;
    dimGrid.z = 1; 
   
    //unsigned int last_element = in_d[num_elements - 1]; 
    printf("radix <bx,gx>=%d,%d\n",dimThreadBlock.x, dimGrid.x);
    //radix<<<dimGrid, dimThreadBlock>>>(in_d, out_d, num_elements);
    /*for (int i =0;i<32;i++) {
        splitGPU<<<dimGrid, dimThreadBlock>>>(in_d,out_d,num_elements,i);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel:splitGPU");

        preScan(out_scan_d, out_d, num_elements);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel");

        indexDefine<<<dimGrid, dimThreadBlock>>>(out_scan_d, out_d, num_elements, last_element);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel");

        scatterElements<<<dimGrid, dimThreadBlock>>>(in_d, out_scan_d, out_d, num_elements);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel");

        // swap pointers
        temp = in_d;
        in_d = out_d;
        out_d = temp;
    }*/
}
