#ifndef IMAGER_H
#define IMAGER_H

// includes
#include "helper.h"
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>

// defs

#define COLOR_MIN 0.f
#define COLOR_MAX 255.f
#define DARKENING_FACTOR .5f
//#define GRAYSCALE_OUTPUT

//#define CONVOLUTION_CONSTANT
  // best with s=.6 a=.002 b=.0006 IMO
//#define FEATURE_DETECTION
//#define ISOTROPIC_DIFFUSION
#define ANISOTROPIC_DIFFUSION

//#define G_1
#define G_MAX
//#define G_EXP


#ifdef FEATURE_DETECTION
#define ROBUST_DERIVATIVE
#endif

#define IDX2(x, y, w) (x)+((y)*w)
#define IDX3(x, y, c, w, h) (x)+((y)*w)+((c)*w*h)

//#define IDX3(x, y, c, w, nc) c+(x*nc)+(y*w*nc)
// using
using namespace std;

// consts
const int KERNEL_RADIUS_MAX = 30;
__constant__ float constKernel[(KERNEL_RADIUS_MAX*2 + 1) * (KERNEL_RADIUS_MAX*2 + 1)];

// globals

// function declarations
__host__ __device__ float g_1(float val);
void kernel(float *dst, int r, float s);
void scale(float *src, float *dst, int n);
__host__ __device__ float get_mat_val(const float *src, int x, int y, int c, int w, int h);
__global__ void conv_device_constant(float *src, float *dst, int w, int h, int r);
__global__ void gradient(float *src, float *dstX,  float *dstY, int w, int h);
__global__ void divergence(float *srcX, float *srcY, float *dst, int w, int h);
__global__ void pointwise_product(float *srcA, float *srcB, float *dst, int w, int h, int nc);
void showSizeableImage(string title, const cv::Mat &mat, int x, int y);

__global__ void feature_detect(float *src11, float *src12, float *src22, float *dst, int w, int h);
__device__ void eigen_values(float *src, float *res);

// function definitions

__host__ __device__ float g_1(float val)
{
    return 1;
}
__host__ __device__ float g_max(float val, float epsilon)
{
    return fmaxf(val, epsilon);
}
__host__ __device__ float g_exp(float val, float epsilon)
{
    return expf((-1)*(val*val)/epsilon) / epsilon;
}

__host__ __device__ float g_roof(float val, float epsilon)
{
#ifdef G_1
    return g_1(val);
#endif
#ifdef G_MAX
    return g_max(val, epsilon);
#endif
#ifdef G_EXP
    return g_exp(val, epsilon);
#endif
}

__global__ void diffusivity(float *imX,  float *imY, int w, int h, float epsilon)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    if (x<w && y<h)
    {
        imX[idx] *= g_roof(imX[idx], epsilon);
        imY[idx] *= g_roof(imY[idx], epsilon);
    }
}

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
}

__device__ void eigen_values(float *src, float *res, float *ev1, float *ev2)
{
    float a = src[0], b = src[1], c = src[2], d = src[3];
    float t = a+d;
    float dt = (a*d) - (b*c);
    float p = sqrtf(((t*t)/4.f)-dt);
    res[0] = 0.5*t-p;
    res[1] = 0.5*t+p;
    
    if (b==0)
    {
        ev1[0] = ev2[1] = 1;
        ev1[1] = ev2[0] = 0;
    }
    else
    {
        ev1[0] = ev2[0] = b;
        ev1[1] = res[0]-a;
        ev2[1] = res[1]-a;
    }
}

__global__ void eigen_maps(float *src11, float *src12, float *src22, float *dstEVal1,
                            float *dstEVal2, float *dstEVec1, float *dstEVec2, int w,
                            int h, float alpha, float beta)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    float evals[2], ev1[2], ev2[2], mat[4];
    if (x < w && y < h)
    {
        mat[0] = src11[idx];
        mat[1] = mat[2] = src12[idx];
        mat[3] = src22[idx];
        eigen_values(mat, evals, ev1, ev2);
        
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



__global__ void gradient(float *src, float *dstX,  float *dstY, int w, int h)
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

__global__ void divergence(float *srcX, float *srcY, float *dst, int w, int h, float tau)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int yt = threadIdx.y + blockDim.y * blockIdx.y;
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    if (x < w && y < h && x>0 && y>0) dst[idx] += tau*((srcX[idx] - srcX[IDX3(x-1,y,c,w,h)]) + (srcY[idx] - srcY[IDX3(x,y-1,c,w,h)]));
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



void showSizeableImage(string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_NORMAL);
    cvMoveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}



#endif
