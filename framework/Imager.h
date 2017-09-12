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

#define IDX(x, y, c, w, nc) ((x+(y*w))*nc) + c

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
    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            p = (float)((a*a)+(b*b))/(float)(2*s*s);
            dst[(a+r)+(b+r)*w] = c*exp((-1)*p);
        }
    }
    
    cv::Mat ker(w, w, CV_32FC1);
    convert_layered_to_mat(ker, dst);
    cv::normalize(ker, ker);
    convert_mat_to_layered (dst, ker);
}

void scale(float *src, float *dst, int n)
{
    float minV = 1, maxV = 0;
    for (int i = 0; i < n; ++i)
    {
        if (src[i] > maxV) maxV = src[i];
        if (src[i] < minV) minV = src[i];
    }
    for (int i = 0; i < n; ++i)
    {
        dst[i] = (src[i] - minV) / (maxV - minV);
    }
}

__host__ __device__ float get_mat_val(const float *src, int x, int y, int c, int w, int h, int nc)
{
    if (x<0)
    {
        if (y<0)
        {
            return src[IDX(0, 0, c, w, nc)];
        } 
        else if (y>=h)
        {
            return src[IDX(0, h-1, c, w, nc)];
        }
        else
        {
            return src[IDX(0, y, c, w, nc)];
        }
    }
    else if (x>w)
    {
        if (y<0)
        {
            return src[IDX(w-1, 0, c, w, nc)];
        } 
        else if (y>=h)
        {
            return src[IDX(w-1, h-1, c, w, nc)];
        }
        else
        {
            return src[IDX(w-1, y, c, w, nc)];
        }
    }
    else
    {
        if (y<0)
        {
            return src[IDX(x, 0, c, w, nc)];
        } 
        else if (y>=h)
        {
            return src[IDX(x, h-1, c, w, nc)];
        }
        else
        {
            return src[IDX(x, y, c, w, nc)];
        }
    }
}

/**
* params  src - source, dst - destination
*         k - kernel, w - source width
*         h - source height, nc - number of channels
*         r - kernel radius(2r+1 X 2r+1)
*/
void conv_host(float *src, float *dst, float *k, int w, int h, int nc, int r)
{
    int kerW = (2*r) + 1;
    for (int c = 0; c < nc; ++c)
    {
        for (int x = 0; x < w; ++x)
        {
            for (int y = 0; y < h; ++y)
            {
                dst[x + w*y] = 0;
                for (int a = -r; a <= r; ++a)
                {
                    for (int b = -r; b <= r; ++b)
                    {
                        //cout << "x:" << x << ", y:" << y << ", c:" << c  << "a:" << a << ", b:" << b << endl;
                        dst[((x + w*y)*nc) + c] += k[(a+r)+(b+r)*kerW]*get_mat_val(src, x-a, y-b, c, w, h, nc);
                    }
                }
            }
        }
    }
}

/**
* params  src - source, dst - destination
*         k - kernel, w - source width
*         h - source height, nc - number of channels
*         r - kernel radius(2r+1 X 2r+1)
*/
__global__ void conv_device(float *src, float *dst, float *k, int w, int h, int nc, int r)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    int c = idx%3;
    size_t fIdx = (idx*nc) + c; //derive linear index
    int kerW = (2*r) + 1;
    //dst[idx] = 0;
    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            if (x < w && y < h) dst[fIdx] += k[(a+r)+(b+r)*kerW]*get_mat_val(src, x-a, y-b, c, w, h, nc);
        }
    }
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


void showSizeableImage(string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_NORMAL);
    cvMoveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}



#endif
