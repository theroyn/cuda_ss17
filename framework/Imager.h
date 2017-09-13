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
//#define CONVOLUTION
//#define CONVOLUTION_SHARED
//#define CONVOLUTION_TEXTURE
#define CONVOLUTION_CONSTANT

#define IDX(x, y, c, w, nc) ((x+(y*w))*nc) + c
#define IDX2(x, y, w) (x)+((y)*w)
#define IDX3(x, y, c, w, h) (x)+((y)*w)+((c)*w*h)

//#define IDX3(x, y, c, w, nc) c+(x*nc)+(y*w*nc)
// using
using namespace std;

// consts
const int KERNEL_RADIUS_MAX = 20;
__constant__ float constKernel[(KERNEL_RADIUS_MAX*2 + 1) * (KERNEL_RADIUS_MAX*2 + 1)];

// globals
texture<float,2,cudaReadModeElementType> texRef;

// function declarations
void kernel(float *dst, int r, float s);
void scale(float *src, float *dst, int n);
__host__ __device__ float get_mat_val(const float *src, int x, int y, int c, int w, int h);
void conv_host(float *src, float *dst, float *k, int w, int h, int nc, int r);
__global__ void conv_device(float *src, float *dst, float *k, int w, int h, int r);
__global__ void conv_device_constant(float *src, float *dst, int w, int h, int r);
__global__ void conv_device_shared(float *src, float *dst, float *k, int w, int h, int r, int smw, int smh);
__global__ void conv_device_texture(float *src, float *dst, float *k, int w, int h, int r);
void gamma_correct_host(float *src, float *dst, int w, int h, int nc, float g);
__global__ void gamma_correct_device(float *src, float *dst, float g, int w, int h, int nc);
__global__ void gradient(float *src, float *dstX,  float *dstY, int w, int h, int nc);
__global__ void divergence(float *srcX, float *srcY, float *dst, int w, int h, int nc);
__global__ void l2_norm(float *src, float *dst, int w, int h, int nc);
void showSizeableImage(string title, const cv::Mat &mat, int x, int y);


// function definitions
void kernel(float *dst, int r, float s)
{
    int w = (2*r)+1;
    float c = 1.f/(float)(2*M_PI*s*s), p, sum = 0.f, add;
    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            p = (float)((a*a)+(b*b))/(float)(2*s*s);
            add = c*exp((-1)*p);
            dst[(a+r)+(b+r)*w] = add;
            sum += add;
        }
    }
    cout << "ker sum:" << sum << endl;
    for (int i = 0; i < (2*r+1)*(2*r+1); ++i)
    {
        dst[i] /= sum;
    }
    
    /**cv::Mat ker(w, w, CV_32FC1), kerTmp(w, w, CV_32FC1);
    convert_layered_to_mat(ker, dst);
    cv::normalize(ker, kerTmp, 1, 0, cv::NORM_L1);
    convert_mat_to_layered (dst, kerTmp);*/
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

__host__ __device__ float get_mat_val(const float *src, int x, int y, int c, int w, int h)
{
    int xt, yt;
    if (x<0) 
    {
        xt = 0;
    }
    else if (x>=w)
    {
        xt = w-1;
    }
    else
    {
        xt = x;
    }

    if (y<0) 
    {
        yt = 0;
    }
    else if (y>=h)
    {
        yt = h-1;
    }
    else
    {
        yt = y;
    }
    return src[IDX3(xt, yt, c, w, h)];
}

/**
* params  src - source, dst - destination
*         k - kernel, w - source width
*         h - source height, nc - number of channels
*         r - kernel radius(2r+1 X 2r+1)
*/
void conv_host(float *src, float *dst, float *k, int w, int h, int nc, int r)
{
    int kerW = (2*r) + 1, idx;
    for (int c = 0; c < nc; ++c)
    {
        for (int x = 0; x < w; ++x)
        {
            for (int y = 0; y < h; ++y)
            {
                idx = IDX3(x,y,c,w,h);
                dst[idx] = 0;
                for (int a = -r; a <= r; ++a)
                {
                    for (int b = -r; b <= r; ++b)
                    {
                        //cout << "x:" << x << ", y:" << y << ", c:" << c  << "a:" << a << ", b:" << b << endl;
                        dst[idx] += k[(a+r)+((b+r)*kerW)]*get_mat_val(src, x-a, y-b, c, w, h);
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
__global__ void conv_device(float *src, float *dst, float *k, int w, int h, int r)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    int kerW = (2*r) + 1;

    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            if (x < w && y < h) dst[idx] += k[(a+r)+((b+r)*kerW)]*get_mat_val(src, x-a, y-b, c, w, h);
        }
    }
}

__global__ void conv_device_constant(float *src, float *dst, int w, int h, int r)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    int kerW = (2*r) + 1;

    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            if (x < w && y < h) dst[idx] += constKernel[(a+r)+((b+r)*kerW)]*get_mat_val(src, x-a, y-b, c, w, h);
        }
    }
}

__global__ void conv_device_texture(float *dst, float *k, int w, int h, int r)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    int kerW = (2*r) + 1;
    float val;

    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            val = tex2D(texRef, (x-a)+0.5f, (yt-a)+0.5f);
            if (x < w && y < h) dst[idx] += k[(a+r)+((b+r)*kerW)]*val;
        }
    }
}

__global__ void conv_device_shared(float *src, float *dst, float *k, int w, int h, int r, int smw, int smh)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    int kerW = (2*r) + 1;

    extern __shared__ float sm[];
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bw = blockDim.x, bh = blockDim.y;
    int mx0 = bw*blockIdx.x - r, my0t = bh*blockIdx.y - r, mx, my, mc;
    int my0 = my0t%h;

    if (x < w && y < h)
    {
        for (int i=(tidx+(tidy*bw)); i < smw*smh; i += bw*bh)
        {
            mx = mx0 + (i%smw);
            my = (my0t + (i%smh))%h;
            sm[i] = get_mat_val(src, mx, my, c, w, h);
        }
    }
    __syncthreads();

    for (int a = -r; a <= r; ++a)
    {
        for (int b = -r; b <= r; ++b)
        {
            if (x < w && y < h) dst[idx] += k[(a+r)+((b+r)*kerW)]*sm[(tidx+r-a) + ((tidy+r-b)*smw)];
        }
    }
}

void gamma_correct_host(float *src, float *dst, int w, int h, int nc, float g)
{
    for (int i = 0; i < w*h*nc; ++i)
    {
        dst[i] = pow(src[i], g);
    }
}

__global__ void gamma_correct_device(float *src, float *dst, float g, int w, int h, int nc)
{

    int xt = threadIdx.x + blockDim.x * blockIdx.x;
    int x = xt/nc;
    int c = xt%nc;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = IDX3(x,y,c,w,h);
    if (x < w && y < h) dst[idx] = powf(src[idx], g);
}

__global__ void gradient(float *src, float *dstX,  float *dstY, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    if (x+1 < w && y < h) dstX[idx] = src[IDX3(x+1,y,c,w,h)] - src[idx];
    if (x < w && y+1 < h) dstY[idx] = src[IDX3(x,y+1,c,w,h)] - src[idx];
}

__global__ void divergence(float *srcX, float *srcY, float *dst, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    if (x < w && y < h && x>0 && y>0) dst[idx] = (srcX[idx] - srcX[IDX3(x-1,y,c,w,h)]) + (srcY[idx] - srcY[IDX3(x,y-1,c,w,h)]);
}

__global__ void l2_norm(float *src, float *dst, int w, int h, int nc)
{
    /**int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = x + w*y; //derive linear index
    int c = idx%nc;
    size_t idx2 = (c + nc*x) +w*y; */

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = IDX2(x,y,w);//x+y*w
    if (x < w && y < h)
    {
        /**for (int c=0; c<nc; ++c) dst[idx] += src[IDX3(x,y,c,w,h)];
        dst[idx] = sqrtf(dst[idx]);*/
        dst[idx] = src[IDX3(x,y,0,w,h)];
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
