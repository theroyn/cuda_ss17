#ifndef IMAGER_H
#define IMAGER_H

// includes
#include "helper.h"
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>

// defs

#define TMP 0
#define COLOR_MIN 0.f
#define COLOR_MAX 255.f
#define DARKENING_FACTOR .5f

//#define GAMMA
//#define GRADIENT
//#define DIVERGENCE
//#define L2
#define LAPLACIAN_NORM
//#define CONVOLUTION
//#define CONVOLUTION_SHARED
//#define CONVOLUTION_TEXTURE
//#define CONVOLUTION_CONSTANT
//#define STRUCTURE_TENSOR
// best with s=.6 a=.002 b=.0006 IMO
//#define FEATURE_DETECTION
#if defined(STRUCTURE_TENSOR) || defined(FEATURE_DETECTION)
#define ROBUST_DERIVATIVE
#endif

#define IDX(x, y, c, w, nc) ((x+(y*w))*nc) + c
#define IDX2(x, y, w) (x)+((y)*(w))
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
__global__ void pointwise_product(float *srcA, float *srcB, float *dst, int w, int h, int nc);
__global__ void l2_norm(float *src, float *dst, int w, int h, int nc);
void showSizeableImage(string title, const cv::Mat &mat, int x, int y);

__global__ void feature_detect(float *src11, float *src12, float *src22, float *dst, int w, int h);
__device__ void eigen_values(float *src, float *res);

// function definitions

__global__ void feature_detect(float *src, float *src11, float *src12,
                                float *src22, float *dst, int w, int h, float alpha, float beta)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = IDX2(x, y, w);
    float eigenvals[2], tensor[4];
    if (x<w && y<h)
    {
        tensor[0] = src11[idx];
        tensor[1] = tensor[2] = src12[idx];
        tensor[3] = src22[idx];
        eigen_values(tensor, eigenvals);
        if (alpha<=eigenvals[0])//red= 255 - ((c+2)/3 * 255)
        {
            dst[IDX3(x, y, 0, w, h)] = COLOR_MAX;
            dst[IDX3(x, y, 1, w, h)] = COLOR_MIN;
            dst[IDX3(x, y, 2, w, h)] = COLOR_MIN;
        }
        else if (beta>=eigenvals[0] && alpha<=eigenvals[1])//yellow = ((4-c)/3)*255 assuming beta<alpha
        {
            dst[IDX3(x, y, 0, w, h)] = COLOR_MAX;
            dst[IDX3(x, y, 1, w, h)] = COLOR_MAX;
            dst[IDX3(x, y, 2, w, h)] = COLOR_MIN;
        }
        else
        {
            dst[IDX3(x, y, 0, w, h)] = src[IDX3(x, y, 0, w, h)]*DARKENING_FACTOR;
            dst[IDX3(x, y, 1, w, h)] = src[IDX3(x, y, 1, w, h)]*DARKENING_FACTOR;
            dst[IDX3(x, y, 2, w, h)] = src[IDX3(x, y, 2, w, h)]*DARKENING_FACTOR;
        }
    }
}

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

__device__ void eigen_values(float *src, float *res)
{
    float t = src[0]+src[3];
    float d = (src[0]*src[3]) - (src[1]*src[2]);
    float p = sqrtf(((t*t)/4.f)-d);
    res[0] = 0.5*t-p;
    res[1] = 0.5*t+p;
    /**float pt = ((t*t)/4.f)-d, p;
    if (pt>=0)
    {
        p = sqrtf(pt);
        res[0] = 0.5*t-p;
        res[1] = 0.5*t+p;
        return 1;
    }
    return 0;*/
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

    /**int xt = threadIdx.x + blockDim.x * blockIdx.x;
    int x = xt/nc;
    int c = xt%nc;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = IDX3(x,y,c,w,h);
    if (x < w && y < h) dst[idx] = powf(src[idx], g);*/

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
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
#ifdef ROBUST_DERIVATIVE
    float p1, p2;
    if (x+1 < w && y+1 < h)
    {
        p1 = 3*src[IDX3(x+1,y+1,c,w,h)] + 10*src[IDX3(x+1,y,c,w,h)] + 3*src[IDX3(x+1,y-1,c,w,h)];
        p2 = 3*src[IDX3(x-1,y+1,c,w,h)] + 10*src[IDX3(x-1,y,c,w,h)] + 3*src[IDX3(x-1,y-1,c,w,h)];
        dstX[idx] = 0.03125f*(p1 - p2);
        p1 = 3*src[IDX3(x+1,y+1,c,w,h)] + 10*src[IDX3(x,y+1,c,w,h)] + 3*src[IDX3(x-1,y+1,c,w,h)];
        p2 = 3*src[IDX3(x+1,y-1,c,w,h)] + 10*src[IDX3(x,y-1,c,w,h)] + 3*src[IDX3(x-1,y-1,c,w,h)];
        dstY[idx] = 0.03125f*(p1 - p2); // (1/32)*...
    }
#else
    if (x+1 < w && y < h) dstX[idx] = src[IDX3(x+1,y,c,w,h)] - src[idx];
    if (x < w && y+1 < h) dstY[idx] = src[IDX3(x,y+1,c,w,h)] - src[idx];
#endif
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

__global__ void pointwise_product(float *srcA, float *srcB, float *dst, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t idx = IDX2(x,y,w);//x+y*w
    if (x < w && y < h)
    {
        for (int c=0; c<nc; ++c) dst[idx] += srcA[IDX3(x,y,c,w,h)]*srcB[IDX3(x,y,c,w,h)];
    }
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
        dst[idx] = 0;
        for (int c=0; c<nc; ++c) dst[idx] += (src[IDX3(x,y,c,w,h)]*src[IDX3(x,y,c,w,h)]);
        dst[idx] = sqrtf(dst[idx]);
        //dst[idx] = src[IDX3(x,y,0,w,h)];
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
