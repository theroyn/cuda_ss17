// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__device__ void square_array(float *a, int n)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx<n) a[idx] = a[idx]*a[idx];
}

__global__ void square_array_wrapper(float *a, int n)
{
    square_array(a, n);
}


int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    for(int i=0; i<n; i++) a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    


    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;

    // copy to device
    float *d_a;
    size_t nbytes = (size_t)(n)*sizeof(int);
    cudaMalloc(&d_a, nbytes); CUDA_CHECK;
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice); CUDA_CHECK;
    
    // launch kernel
    dim3 block = dim3(128,1,1);
    // dim3 grid = dim3((n + block.x –1) / block.x, 1, 1);
    dim3 grid = dim3((n+block.x-1)/block.x,1,1);

    square_array_wrapper<<<grid, block>>>(d_a, n);

    // copy to host and deallocate
    cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(d_a); CUDA_CHECK;

    
    // ###
    // ### TODO: Implement the "square array" operation on the GPU and store the result in "a"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"


    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
}



