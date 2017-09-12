#ifndef IMAGER_H
#define IMAGER_H

// includes
#include "helper.h"
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>

// defs
#define TMP 0
//#define GAMMA
//#define GRADIENT
//#define DIVERGENCE
//#define L2
//#define LAPLACIAN_NORM
#define CONVOLUTION

// using
using namespace std;

// consts

// function declarations
void gamma_correct_host(float *src, float *dst, int w, int h, int nc, float g);
int get_tmp();
void cuda_check(string file, int line);
void kernel(float * dst, int r);
__global__ void gamma_correct_device(float *src, float *dst, float g, int w, int h, int n);
__device__ void derive_x(float *src, float *dst, int w, int h, int n);
__device__ void derive_y(float *src, float *dst, int w, int h, int n);
__global__ void gradient(float *src, float *dst, int w, int h, int n);

// function definitions
void kernel(float *dst, int r, float s)
{
    int w = (2*r)+1;
    float c = 1.f/(float)(2*M_PI*s*s), p;
    cout << "c:" << c << endl;
    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            p = (float)((a*a)+(b*b))/(float)(2*s*s);
            cout << "a:" << a << ", b:" << b << ", p:" << p << endl;
            dst[(a+r)+(b+r)*w] = c*exp((-1)*p);
        }
    }
    
    cv::Mat ker(w, w, CV_32FC1);
    convert_layered_to_mat(ker, dst);
    cv::normalize(ker, ker);
    convert_mat_to_layered (dst, ker);
}

int get_tmp()
{
    return TMP;
}

void gamma_correct_host(float *src, float *dst, int w, int h, int nc, float g)
{
    for (int i = 0; i < w*h*nc; ++i)
    {
        dst[i] = pow(src[i], g);
    }
}

__global__ void gamma_correct_device(float *src, float *dst, float g, int w, int h, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    if (x < (n/h) && y < (n/w)) dst[idx] = powf(src[idx], g);
}

/**__device__ void derive_x(float *src, float *dst, int w, int h, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    if (x+1 < (n/h) && y < (n/w)) dst[idx] = src[idx+1] - src[idx];
}*/

/**__device__ void derive_y(float *src, float *dst, int w, int h, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    if (x < (n/h) && y+1 < (n/w)) dst[idx] = src[idx+w] - src[idx];
}*/

__global__ void gradient(float *src, float *dstX,  float *dstY, int w, int h, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    if (x+1 < (n/h) && y < (n/w)) dstX[idx] = src[idx+1] - src[idx];
    if (x < (n/h) && y+1 < (n/w)) dstY[idx] = src[idx+w] - src[idx];
}

__global__ void divergence(float *srcX, float *srcY, float *dst, int w, int h, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    if (x < (n/h) && y < (n/w) && x>0 && y>0) dst[idx] = (srcX[idx] - srcX[idx-1]) + (srcY[idx] - srcY[idx-w]);
}

__global__ void l2_norm(float *src, float *dst, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    if (x < w && y < h)
    {
        for (int c=0; c<nc; ++c) dst[idx] += src[idx*nc + c];
        dst[idx] = sqrtf(dst[idx]);
    } 
}





#endif
