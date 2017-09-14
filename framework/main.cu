// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include "helper.h"
#include <iostream>
#include "Imager.h"
#include <stdlib.h> 

using namespace std;

// uncomment to use the camera
//#define CAMERA






int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed
    float gamma = 1.f, sigma = 1.f, factVal = 1.f;
    getParam("g", gamma, argc, argv);
    cout << "gamma: " << gamma << endl;
    getParam("s", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;
    getParam("f", factVal, argc, argv);
    cout << "factor: " << factVal << endl;

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
#if defined(L2) || defined(LAPLACIAN_NORM)
    cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
#else
    cout << "mIn.channels():" << mIn.channels() << endl;
    cout << "mIn.type():" << mIn.type() << endl;
    cout << "CV_32FC1:" << CV_32FC1 << endl;
    cout << "CV_32FC3:" << CV_32FC3 << endl;
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
#endif
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    // ### Define your own output images here as needed
#ifdef STRUCTURE_TENSOR
    cv::Mat mOut11(h,w,CV_32FC1); 
    cv::Mat mOut12(h,w,CV_32FC1); 
    cv::Mat mOut22(h,w,CV_32FC1); 
#endif
    



    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];
    //std::unique_ptr<float[]> imgIn(new float[(size_t)w*h*nc]);

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    //float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // gpu vars init
#ifdef GAMMA
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    //std::unique_ptr<float[]> imgOut(new float[(size_t)nO]);
    float *d_imgIn, *d_imgOut;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#ifdef GRADIENT
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    //std::unique_ptr<float[]> imgOut(new float[(size_t)nO]);
    float *d_imgIn, *d_imgOut, *d_gX, *d_gY;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gX, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_gY, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#ifdef DIVERGENCE
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    //std::unique_ptr<float[]> imgOut(new float[(size_t)nO]);
    float *d_imgIn, *d_imgOut, *d_gX, *d_gY;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gX, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gY, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#ifdef L2
    int nI = w*h*nc;
    int nO = w*h;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    //std::unique_ptr<float[]> imgOut(new float[(size_t)nO]);
    float *d_imgIn, *d_imgOut;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#ifdef LAPLACIAN_NORM
    int nI = w*h*nc;
    int nO = w*h;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    //std::unique_ptr<float[]> imgOut(new float[(size_t)nO]);
    float *d_imgIn, *d_imgOut, *d_gX, *d_gY, *d_divOut;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gX, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gY, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_divOut, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#if defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) || defined(CONVOLUTION_TEXTURE)
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    //std::unique_ptr<float[]> imgOut(new float[(size_t)nO]);
    float *d_imgIn, *d_imgOut, *k, *d_k;
    int r = ceil(sigma * 3);
    int d = (2*r)+1;
    k = new float[(size_t)(d * d)];
    kernel(k, r, sigma);
    cv::Mat mKer(d, d, CV_32FC1);
    float *kt = new float[(size_t)(d * d)];
    scale(k, kt, d*d);
    convert_layered_to_mat(mKer, kt);
    cout << "mKer: " << mKer << endl;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
    int nK = d*d; // kernel
    size_t nbytesK = (size_t)(nK)*sizeof(float);
    cudaMalloc(&d_k, nbytesK); CUDA_CHECK;
    cudaMemcpy(d_k, k, nbytesK, cudaMemcpyHostToDevice); CUDA_CHECK;
#endif
#ifdef CONVOLUTION_CONSTANT
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    float *d_imgIn, *d_imgOut, *k;

    int r = ceil(sigma * 3);
    int d = (2*r)+1;
    k = new float[(size_t)(d * d)];
    kernel(k, r, sigma);
    cv::Mat mKer(d, d, CV_32FC1);
    float *kt = new float[(size_t)(d * d)];
    scale(k, kt, d*d);
    convert_layered_to_mat(mKer, kt);
    cout << "mKer: " << mKer << endl;

    cudaMemcpyToSymbol(constKernel, k, (size_t)(d * d)*sizeof(float)); CUDA_CHECK;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#ifdef STRUCTURE_TENSOR
    int nI = w*h*nc;
    int nO = w*h;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    float *imgOut11 = (float *) malloc (nbytesO);
    float *imgOut12 = (float *) malloc (nbytesO);
    float *imgOut22 = (float *) malloc (nbytesO);
    float *d_imgIn, *d_s, *k, *d_gX, *d_gY, *d_m11, *d_m12, *d_m22, *d_imgOut11, *d_imgOut12, *d_imgOut22;

    int r = ceil(sigma * 3);
    int d = (2*r)+1;
    k = new float[(size_t)(d * d)];
    kernel(k, r, sigma);
    cudaMemcpyToSymbol(constKernel, k, (size_t)(d * d)*sizeof(float)); CUDA_CHECK;


    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_s, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gX, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gY, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_m11, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_m12, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_m22, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_imgOut11, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_imgOut12, nbytesO); CUDA_CHECK;
    cudaMalloc(&d_imgOut22, nbytesO); CUDA_CHECK;
#endif


    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
        // Get camera image
        camera >> mIn;
        // convert to float representation (opencv loads image values as single bytes by default)
        mIn.convertTo(mIn,CV_32F);
        // convert range of each channel to [0,1] (opencv default is [0,255])
        mIn /= 255.f;
#endif
#ifdef L2
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif
#ifdef GRADIENT
        cudaMemset(d_gX, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_gY, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif
#ifdef DIVERGENCE
        cudaMemset(d_gX, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_gY, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif
#ifdef LAPLACIAN_NORM
        cudaMemset(d_gX, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_gY, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_divOut, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif 
#if defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) || defined(CONVOLUTION_TEXTURE) || defined(CONVOLUTION_CONSTANT)
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif
#ifdef STRUCTURE_TENSOR
        cudaMemset(d_s, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_gX, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_gY, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_m11, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_m12, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_m22, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_imgOut11, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_imgOut12, 0, nbytesO); CUDA_CHECK;
        cudaMemset(d_imgOut22, 0, nbytesO); CUDA_CHECK;
#endif

        // Init raw input image array
        // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
        // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
        // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
        convert_mat_to_layered (imgIn, mIn);




        // copy data from host to device
#if defined(GAMMA) || defined(GRADIENT) || defined(DIVERGENCE) || defined(L2) \
|| defined(LAPLACIAN_NORM) || defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) \
|| defined(CONVOLUTION_TEXTURE) || defined(CONVOLUTION_CONSTANT) || defined(STRUCTURE_TENSOR)
        cudaMemcpy(d_imgIn, imgIn, nbytesI, cudaMemcpyHostToDevice); CUDA_CHECK;
        //memset(imgOut, 0, nbytesO);
#endif
#ifdef CONVOLUTION_TEXTURE
        texRef.addressMode[0] = cudaAddressModeClamp; // clamp x to border
        texRef.addressMode[1] = cudaAddressModeClamp; // clamp y to border
        texRef.filterMode = cudaFilterModeLinear; // linear interpolation
        texRef.normalized = false; // access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaBindTexture2D(NULL, &texRef, d_imgIn, &desc, w, h*nc, w*sizeof(d_imgIn[0])); CUDA_CHECK;
#endif
        Timer timer; timer.start();
        // ###
        // ###
        // ### TODO: Main computation
        // ###
        // ###

        for (int i = 0; i < repeats; ++i)
        {
#ifdef GAMMA
            //gamma_correct_host(imgIn, imgOut, w, h, nc, gamma);
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            gamma_correct_device<<<grid, block>>>(d_imgIn, d_imgOut, gamma, w, h, nc);
#endif
#ifdef GRADIENT
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            gradient<<<grid, block>>>(d_imgIn, d_gX, d_imgOut, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;//d_imgOut, d_gX, d_gY
#endif
#ifdef DIVERGENCE
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            gradient<<<grid, block>>>(d_imgIn, d_gX, d_gY, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;
            divergence<<<grid, block>>>(d_gX, d_gY, d_imgOut, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;
#endif
#ifdef L2
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
            l2_norm<<<grid, block>>>(d_imgIn, d_imgOut, w, h, nc);
#endif
#ifdef LAPLACIAN_NORM
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            gradient<<<grid, block>>>(d_imgIn, d_gX, d_gY, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;
            divergence<<<grid, block>>>(d_gX, d_gY, d_divOut, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;
            grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
            l2_norm<<<grid, block>>>(d_divOut, d_imgOut, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;
#endif
#ifdef CONVOLUTION
            //conv_host(imgIn, imgOut, k, w, h, nc, r);
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device<<<grid, block>>>(d_imgIn, d_imgOut, d_k, w, h, r);
#endif
#ifdef CONVOLUTION_CONSTANT
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_constant<<<grid, block>>>(d_imgIn, d_imgOut, w, h, r);
#endif
#ifdef CONVOLUTION_TEXTURE
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_texture<<<grid, block>>>(d_imgOut, d_k, w, h, r);
#endif
#ifdef CONVOLUTION_SHARED
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            int smw = block.x + 2*r;
            int smh = block.y + 2*r;
            size_t smbytes = smw*smh*sizeof(float);
            conv_device_shared<<<grid, block, smbytes>>>(d_imgIn, d_imgOut, d_k, w, h, r, smw, smh);
            cudaDeviceSynchronize();  CUDA_CHECK;
#endif
#ifdef ROBUST_DERIVATIVE
            cout << "rub" << endl;
#else
            cout << "no rub" << endl;
#endif
#ifdef STRUCTURE_TENSOR
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_constant<<<grid, block>>>(d_imgIn, d_s, w, h, r); // Compute S = G σ ∗ u
            cudaDeviceSynchronize();  CUDA_CHECK;
            gradient<<<grid, block>>>(d_s, d_gX, d_gY, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;
            /**cudaMemcpy(imgOut, d_gX, nbytesO, cudaMemcpyDeviceToHost); CUDA_CHECK;
            convert_layered_to_mat(mOut, imgOut);
            cout << "mOutX" << mOut(cv::Range(0,15), cv::Range(0,15)) << endl;
            showSizeableImage("OutputX", mOut, 100+w+40, 100);*/

            grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
            // d_mxx are grayscale
            pointwise_product<<<grid, block>>>(d_gX, d_gX, d_m11, w, h, nc);
            pointwise_product<<<grid, block>>>(d_gX, d_gY, d_m12, w, h, nc);
            pointwise_product<<<grid, block>>>(d_gY, d_gY, d_m22, w, h, nc);
            
            conv_device_constant<<<grid, block>>>(d_m11, d_imgOut11, w, h, r);
            conv_device_constant<<<grid, block>>>(d_m12, d_imgOut12, w, h, r);
            conv_device_constant<<<grid, block>>>(d_m22, d_imgOut22, w, h, r);
#endif
        }
        cudaDeviceSynchronize();  CUDA_CHECK;
        timer.end();  float t = timer.get() / (float) repeats;  // elapsed time in seconds
        cout << "time: " << t*1000 << " ms" << endl;

        // copy data from device to host
#if defined(GAMMA) || defined(GRADIENT) || defined(DIVERGENCE) || \
defined(L2) || defined(LAPLACIAN_NORM) || defined(CONVOLUTION) \
|| defined(CONVOLUTION_SHARED) || defined(CONVOLUTION_TEXTURE) || defined(CONVOLUTION_CONSTANT)
        cudaMemcpy(imgOut, d_imgOut, nbytesO, cudaMemcpyDeviceToHost); CUDA_CHECK;
#endif
#ifdef STRUCTURE_TENSOR
        cudaMemcpy(imgOut11, d_imgOut11, nbytesO, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(imgOut12, d_imgOut12, nbytesO, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(imgOut22, d_imgOut22, nbytesO, cudaMemcpyDeviceToHost); CUDA_CHECK;
#endif

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

#ifndef STRUCTURE_TENSOR
        // show output image: first convert to interleaved opencv format from the layered raw array
        convert_layered_to_mat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);
#else
        convert_layered_to_mat(mOut11, imgOut11);
        convert_layered_to_mat(mOut12, imgOut12);
        convert_layered_to_mat(mOut22, imgOut22);
        mOut11 *= factVal;
        mOut12 *= factVal;
        mOut22 *= factVal;
        cout << "mOut11" << mOut11(cv::Range(0,5), cv::Range(0,5)) << endl;
        showImage("Output11", mOut11, 100+w+40, 100);
        showImage("Output12", mOut12, 100, 100+h+40);
        showImage("Output22", mOut22, 100+w+40, 100+h+40);
#endif

        // ### Display your own output images here as needed
#if defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) || defined(CONVOLUTION_TEXTURE) || defined(CONVOLUTION_CONSTANT)
        showSizeableImage("Kernel", mKer, 100, 100+h+40);
#endif

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

//all
//imgIn X
//ifdef GAMMA
//imgOut, d_imgIn, *d_imgOut
//ifdef GRADIENT
//imgOut, d_imgIn, *d_imgOut, *d_gX, *d_gY
//ifdef DIVERGENCE
//imgOut, d_imgIn, *d_imgOut, *d_gX, *d_gY;
//ifdef L2
//imgOut, d_imgIn, *d_imgOut
//ifdef LAPLACIAN_NORM
//imgOut, d_imgIn, *d_imgOut, *d_gX, *d_gY, *d_divOut
//if defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) || defined(CONVOLUTION_TEXTURE)
//imgOut, *d_imgIn, *d_imgOut, *k, *d_k, kt
//ifdef CONVOLUTION_CONSTANT
//imgOut, *d_imgIn, *d_imgOut, *k, kt
    cudaDeviceSynchronize();  CUDA_CHECK;
    //cpu deallocs
    free(imgIn);
    delete[] imgOut;
#if defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) \
|| defined(CONVOLUTION_TEXTURE) || defined(CONVOLUTION_CONSTANT)
    delete[] k;
    delete[] kt;
#endif
    // gpu vars deallocs
#if defined(GAMMA) || defined(L2) || defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) \
|| defined(CONVOLUTION_TEXTURE) || defined(CONVOLUTION_CONSTANT)
    cout << "nbytesI: " << nbytesI << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
#endif
#ifdef CONVOLUTION_TEXTURE
    cudaUnbindTexture(texRef);
#endif
#ifdef GRADIENT
    cout << "nbytesI: " << nbytesI << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_gX); CUDA_CHECK;
    cudaFree(d_gY); CUDA_CHECK;
#endif
#ifdef DIVERGENCE
    cout << "nbytesI: " << nbytesI << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_gX); CUDA_CHECK;
    cudaFree(d_gY); CUDA_CHECK;
#endif
#ifdef LAPLACIAN_NORM
    cout << "nbytesI: " << nbytesI << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_gX); CUDA_CHECK;
    cudaFree(d_gY); CUDA_CHECK;
    cudaFree(d_divOut); CUDA_CHECK;
#endif
#if defined(CONVOLUTION) || defined(CONVOLUTION_SHARED) || defined(CONVOLUTION_TEXTURE)
    cudaFree(d_k); CUDA_CHECK;
#endif
#ifdef STRUCTURE_TENSOR
// d_gX, *d_gY, *d_m11, *d_m12, *d_m22, *d_imgOut11, *d_imgOut12, *d_imgOut22

    cout << "nbytesI: " << nbytesI << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_gX); CUDA_CHECK;
    cudaFree(d_gY); CUDA_CHECK;
    cudaFree(d_m11); CUDA_CHECK;
    cudaFree(d_m12); CUDA_CHECK;
    cudaFree(d_m22); CUDA_CHECK;
    cudaFree(d_imgOut11); CUDA_CHECK;
    cudaFree(d_imgOut12); CUDA_CHECK;
    cudaFree(d_imgOut22); CUDA_CHECK;

    
    free(imgOut11);
    free(imgOut12);
    free(imgOut22);
#endif


    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    //delete[] imgIn;
    //delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



