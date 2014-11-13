//#include <stdio.h>
#include <iostream>
#include<algorithm>
#include <stdlib.h>
#include <stdint.h>

#include "support.h"
#include "kernel.cu"
#include<vector>
#include<utility>
using namespace std;


bool pair_compare(const pair<short unsigned int, unsigned int>& p1,const pair<short unsigned int, unsigned int>& p2);
int main(int argc, char* argv[])
{
    FILE *fp = fopen("topic-1.txt", "r");
    if (fp == NULL){
        cout<<"Can't read file";
        exit(0);
    }
    cout<<"fp opened";
    char *line = NULL;
    size_t len = 0;
    unsigned int lines = 0;
    unsigned int count = 0;
    char *ln, *nptr;

    unsigned int *transactions = NULL;
    unsigned int *trans_offset = NULL;
    unsigned int *flist = NULL;
    unsigned short *flist_key_16 = NULL;
    unsigned short *flist_key_16_index = NULL;

    unsigned int element_id = 0;
    unsigned int item_name = 0;

    transactions = (unsigned int *) malloc(max_num_of_transaction * max_items_in_transaction * sizeof(unsigned int));
    trans_offset = (unsigned int *) malloc((max_num_of_transaction + 1) * sizeof(unsigned int));
    flist = (unsigned int *) malloc(max_unique_items * sizeof(unsigned int));
    flist_key_16 = (unsigned short*) malloc(max_unique_items * sizeof(unsigned short));
    flist_key_16_index = (unsigned short*) malloc(max_unique_items * sizeof(unsigned short));

    memset(flist_key_16_index, 0xFFFF, max_unique_items * sizeof(unsigned short));

    trans_offset[0] = 0;
    while (getline(&line, &len, fp) != -1 && lines < max_num_of_transaction){
       count = 0;
        ln = strtok(line, " ");
        if (ln != NULL){
                //unsigned int a = (unsigned int) strtoul(ln, NULL, 0);
                item_name = (unsigned int) strtoul(ln, NULL, 0);
                transactions[element_id++] = item_name;
                if (item_name < max_unique_items) flist[item_name] = 0;
                count++;
        }

        while (ln != NULL){
            ln = strtok(NULL, " ");
            if (ln != NULL){
                item_name = (unsigned int) strtoul(ln, &nptr, 0);
                if (strcmp(nptr, ln) != 0){
                    transactions[element_id++] = item_name;
                    if (item_name < max_unique_items) flist[item_name] = 0;
                    count++;
                }
            }
        }

        trans_offset[lines + 1] = trans_offset[lines] + count;
        lines++;
    }
    cout<<"file parsed\n";
    fclose(fp);

    //trans_offset[lines] = NULL;
    //transactions[element_id] = NULL;
    unsigned int num_items_in_transactions = element_id;
    unsigned int num_transactions = lines - 1;

    cout<<"Number of Transactions = "<<num_transactions<<endl;
    cout<<"num_items_in_transactions = "<<num_items_in_transactions<<endl;
    #if TEST_MODE
    for (int i = 0; i < num_transactions; i++){
        int item_ends = 0;
        if (i == (num_transactions - 1)){
            item_ends = num_items_in_transactions;
        }else{
            item_ends = trans_offset[i+1];
        }
        for (int j = trans_offset[i]; j < item_ends; j++)
            cout<<transactions[j]<<" ";
        cout<<endl;
    }
    for (int i = 0; i <= num_transactions; i++) {
       cout<<trans_offset[i]<<","; 
    }
    #endif


    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////// Device Variables Initializations ///////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    Timer timer;
    cudaError_t cuda_ret;

    unsigned int *d_transactions;
    unsigned int *d_trans_offsets;
    unsigned int *d_flist, *d_flist_key_16, *d_flist_key_16_index;
    cudaDeviceProp deviceProp;
    cudaError_t ret;
    cudaGetDeviceProperties(&deviceProp, 0);
    int SM_PER_BLOCK = deviceProp.sharedMemPerBlock;
    int CONST_MEM_GPU = deviceProp.totalConstMem;
    // Allocate device variables ----------------------------------------------

    cout<<"Allocating device variables...";
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&d_transactions, num_items_in_transactions * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_trans_offsets, num_transactions * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_flist, max_unique_items * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_flist_key_16_index, max_unique_items * sizeof(unsigned short));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_flist_key_16, max_unique_items * sizeof(unsigned short));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");


    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    cuda_ret = cudaMemcpy(d_transactions, transactions, num_items_in_transactions * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    // to test
	cuda_ret = cudaMemcpy(d_trans_offsets, trans_offset, num_transactions * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    
    cuda_ret = cudaMemset(d_flist, 0, max_unique_items * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
    
    startTime(&timer);
    cout<<"histogram kernel\n";
    make_flist(d_trans_offsets, d_transactions, d_flist, num_transactions, num_items_in_transactions, SM_PER_BLOCK);
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    startTime(&timer);
    cout<<"copying flist form dev to host\n";
    cuda_ret = cudaMemcpy(flist, d_flist, max_unique_items * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    //now sort the flist
    vector<pair<unsigned short, unsigned int> > v;
    //map<unsigned int, short unsigned int> m;
    cout<<"sorting and pruning flist:\n";
    startTime(&timer);
    for (int i =0; i < max_unique_items;i++) {
        if (flist[i] >= support) {
            v.push_back(pair<unsigned short, unsigned int>(i, flist[i]));
        } 
    }
    // print the vector
    #if TEST_MODE
    cout<<"vector length:"<<v.size()<<endl;
    vector<pair<unsigned short , unsigned int> >::iterator it;
    for (it = v.begin(); it != v.end();it++) {
        cout<<"(key,value)"<<it->first<<","<<it->second<<endl;    
    }
    #endif
    std::sort(v.begin(),v.end(), pair_compare);
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    cout<<"copying sort data to flist and key arrays:\n";
    startTime(&timer);
    vector<pair<unsigned short , unsigned int> >::reverse_iterator itr;
    int i = 0;
    for (itr = v.rbegin(); itr != v.rend();itr++) {
        flist[i] = itr->second;
        flist_key_16[i] = itr->first;
        flist_key_16_index[itr->first] = i;
    #if TEST_MODE
        cout<<"(count,key):"<<flist[i]<<","<<flist_key_16[i]<<endl;
    #endif
        i++;
    }

    #if TEST_MODE
    for (i  = 0; i < max_unique_items; i++) {
        cout<<"(key,key_index_in_flist)"<<i<<","<<flist_key_16_index[i]<<endl;   
    }
    #endif
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    cout<<"constant mem available:"<<CONST_MEM_GPU<<endl;

    cout<<"copy flist and key arrays back to device:\n";
    startTime(&timer);
    cuda_ret = cudaMemcpy(d_flist, flist, max_unique_items * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    
    cuda_ret = cudaMemcpy(d_flist_key_16, flist_key_16, max_unique_items * sizeof(unsigned short),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    
    if (max_unique_items * sizeof(unsigned short) < CONST_MEM_GPU) {
        // keep the index file in constant memory
    #if TEST_MODE
        cout<<"copying to constant mem"<<endl;
    #endif
        cuda_ret = cudaMemcpyToSymbol(dc_flist_key_16_index, flist_key_16_index, max_unique_items * sizeof(unsigned short), 0, cudaMemcpyHostToDevice);
    } else {
    #if TEST_MODE
        cout<<"copying to global mem"<<endl;
    #endif
        cuda_ret = cudaMemcpy(d_flist_key_16_index, flist_key_16_index, max_unique_items * sizeof(unsigned short),
        cudaMemcpyHostToDevice);
    }
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    
    // Free memory ------------------------------------------------------------

    cudaFree(d_trans_offsets);
    cudaFree(d_transactions);
    cudaFree(d_flist);
    cudaFree(d_flist_key_16);
    cudaFree(d_flist_key_16_index);

    free(trans_offset);
    free(transactions);
    free(flist);
    free(flist_key_16);
    free(flist_key_16_index);
    cout<<"program end";
}

bool pair_compare(const pair<short unsigned int, unsigned int>& p1,const pair<short unsigned int, unsigned int>& p2) {
    return p1.second < p2.second;    
}
