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
    float gamma = 1.f, sigma = 1.f, factVal = 1.f, alpha, beta, tau, epsilon, rho;
    int nsteps = 1;
    getParam("s", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;
    getParam("f", factVal, argc, argv);
    cout << "factor: " << factVal << endl;
    getParam("a", alpha, argc, argv);
    cout << "alpha: " << alpha << endl;
    getParam("b", beta, argc, argv);
    cout << "beta: " << beta << endl;
    getParam("t", tau, argc, argv);
    cout << "tau: " << tau << endl;;
    getParam("e", epsilon, argc, argv);
    cout << "epsilon: " << epsilon << endl;
    getParam("r", rho, argc, argv);
    cout << "rho: " << rho << endl;
    getParam("steps", nsteps, argc, argv);
    cout << "steps: " << nsteps << endl;

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
#ifdef GRAYSCALE_OUTPUT
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
    cv::Mat mOut2(h,w,mIn.type());

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
#ifdef FEATURE_DETECTION
    int nI = w*h*nc;
    int nG = w*h;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesG = (size_t)(nG)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    float *d_imgIn, *d_s, *k, *d_gX, *d_gY, *d_m11, *d_m12,
            *d_m22, *d_imgOut11, *d_imgOut12, *d_imgOut22, *d_imgOut;

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
    cudaMalloc(&d_m11, nbytesG); CUDA_CHECK;
    cudaMalloc(&d_m12, nbytesG); CUDA_CHECK;
    cudaMalloc(&d_m22, nbytesG); CUDA_CHECK;
    cudaMalloc(&d_imgOut11, nbytesG); CUDA_CHECK;
    cudaMalloc(&d_imgOut12, nbytesG); CUDA_CHECK;
    cudaMalloc(&d_imgOut22, nbytesG); CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytesO); CUDA_CHECK;
#endif
#ifdef ISOTROPIC_DIFFUSION
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    float *imgOut2 = (float *) malloc (nbytesO);
    float *d_imgIn, *d_gX, *d_gY, *d_s, *k;

    if (tau>=0.25f/g_roof(0, epsilon)) tau = 0.2499f/g_roof(0, epsilon);
    sigma = sqrtf(2*tau*nsteps);
    cout << "final sigma" << sigma << ", final tau" << tau << endl;
    int r = ceil(sigma * 3);
    int d = (2*r)+1;
    k = new float[(size_t)(d * d)];
    kernel(k, r, sigma);
    cv::Mat mKer(d, d, CV_32FC1);
    cudaMemcpyToSymbol(constKernel, k, (size_t)(d * d)*sizeof(float)); CUDA_CHECK;

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gX, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gY, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_s, nbytesI); CUDA_CHECK;
#endif
#ifdef ANISOTROPIC_DIFFUSION
    int nI = w*h*nc;
    int nO = w*h*nc;
    size_t nbytesI = (size_t)(nI)*sizeof(float);
    size_t nbytesO = (size_t)(nO)*sizeof(float);
    float *imgOut = (float *) malloc (nbytesO);
    float *imgOut2 = (float *) malloc (nbytesO);
    float *d_imgIn, *d_gX, *d_gY, *d_s, *k, *kRho;

    if (tau>=0.25f/g_roof(0, epsilon)) tau = 0.2499f/g_roof(0, epsilon);
    sigma = sqrtf(2*tau*nsteps);
    cout << "final sigma" << sigma << ", final tau" << tau << endl;
    int r = ceil(sigma * 3);
    int d = (2*r)+1;
    k = new float[(size_t)(d * d)];
    kernel(k, r, sigma);
    cudaMemcpyToSymbol(constKernel, k, (size_t)(d * d)*sizeof(float)); CUDA_CHECK;

    int rRho = ceil(rho * 3);
    int dRho = (2*r)+1;
    kRho = new float[(size_t)(d * d)];
    kernel(kRho, rRho, sigmaRho);

    // gpu allocs
    cudaMalloc(&d_imgIn, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gX, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_gY, nbytesI); CUDA_CHECK;
    cudaMalloc(&d_s, nbytesI); CUDA_CHECK;
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

#ifdef CONVOLUTION_CONSTANT
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif
#ifdef FEATURE_DETECTION
        cudaMemset(d_s, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_gX, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_gY, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_m11, 0, nbytesG); CUDA_CHECK;
        cudaMemset(d_m12, 0, nbytesG); CUDA_CHECK;
        cudaMemset(d_m22, 0, nbytesG); CUDA_CHECK;
        cudaMemset(d_imgOut11, 0, nbytesG); CUDA_CHECK;
        cudaMemset(d_imgOut12, 0, nbytesG); CUDA_CHECK;
        cudaMemset(d_imgOut22, 0, nbytesG); CUDA_CHECK;
        cudaMemset(d_imgOut, 0, nbytesO); CUDA_CHECK;
#endif
#ifdef ISOTROPIC_DIFFUSION
        cudaMemset(d_s, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_gX, 0, nbytesI); CUDA_CHECK;
        cudaMemset(d_gY, 0, nbytesI); CUDA_CHECK;
#endif

        // Init raw input image array
        // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
        // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
        // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
        convert_mat_to_layered (imgIn, mIn);




        // copy data from host to device
#if defined(CONVOLUTION_CONSTANT) || defined(FEATURE_DETECTION) || defined(ISOTROPIC_DIFFUSION)
        cudaMemcpy(d_imgIn, imgIn, nbytesI, cudaMemcpyHostToDevice); CUDA_CHECK;
        //memset(imgOut, 0, nbytesO);
#endif
        Timer timer; timer.start();

        // Main computation
        for (int i = 0; i < repeats; ++i)
        {

#ifdef CONVOLUTION_CONSTANT
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_constant<<<grid, block>>>(d_imgIn, d_imgOut, w, h, r);
#endif
#ifdef FEATURE_DETECTION
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_constant<<<grid, block>>>(d_imgIn, d_s, w, h, r); // Compute S = G σ ∗ u
            cudaDeviceSynchronize();  CUDA_CHECK;
            gradient<<<grid, block>>>(d_s, d_gX, d_gY, w, h);
            cudaDeviceSynchronize();  CUDA_CHECK;

            grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
            pointwise_product<<<grid, block>>>(d_gX, d_gX, d_m11, w, h, nc);
            pointwise_product<<<grid, block>>>(d_gX, d_gY, d_m12, w, h, nc);
            pointwise_product<<<grid, block>>>(d_gY, d_gY, d_m22, w, h, nc);
            
            conv_device_constant<<<grid, block>>>(d_m11, d_imgOut11, w, h, r);
            conv_device_constant<<<grid, block>>>(d_m12, d_imgOut12, w, h, r);
            conv_device_constant<<<grid, block>>>(d_m22, d_imgOut22, w, h, r);

            feature_detect<<<grid, block>>>(d_imgIn, d_imgOut11, d_imgOut12, d_imgOut22, d_imgOut, w, h, alpha, beta);
#endif
#ifdef ISOTROPIC_DIFFUSION
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_constant<<<grid, block>>>(d_imgIn, d_s, w, h, r); // Compute S = G σ ∗ u
            for (int i=0; i < nsteps; ++i)
            {
                gradient<<<grid, block>>>(d_imgIn, d_gX, d_gY, w, h);
                diffusivity<<<grid, block>>>(d_gX, d_gY, w, h, epsilon);
                divergence<<<grid, block>>>(d_gX, d_gY, d_imgIn, w, h, tau);
            }
            
#endif
#ifdef ANISOTROPIC_DIFFUSION
            dim3 block = dim3(32, 8, 1);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h*nc + block.y - 1) / block.y, 1);
            conv_device_constant<<<grid, block>>>(d_imgIn, d_s, w, h, r); // Compute S = G σ ∗ u
            cudaDeviceSynchronize();  CUDA_CHECK;
            cudaMemcpyToSymbol(constKernel, kRho, (size_t)(dRho * dRho)*sizeof(float)); CUDA_CHECK;
            gradient<<<grid, block>>>(d_s, d_gX, d_gY, w, h, nc);
            cudaDeviceSynchronize();  CUDA_CHECK;

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
#if defined(CONVOLUTION_CONSTANT) || defined(FEATURE_DETECTION)
        cudaMemcpy(imgOut, d_imgOut, nbytesO, cudaMemcpyDeviceToHost); CUDA_CHECK;
#endif
#ifdef ISOTROPIC_DIFFUSION
        //cout << "cudaMemcpy" << endl;
        cudaMemcpy(imgOut, d_imgIn, nbytesI, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(imgOut2, d_s, nbytesI, cudaMemcpyDeviceToHost); CUDA_CHECK;
#endif

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)


        // show output image: first convert to interleaved opencv format from the layered raw array
        convert_layered_to_mat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        // ### Display your own output images here as needed
        convert_layered_to_mat(mOut2, imgOut2);
        showImage("Gaussian", mOut2, 100, 100+h+40);
#ifdef CONVOLUTION_CONSTANT
        showSizeableImage("Kernel", mKer, 100, 100+h+40);
#endif

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

    cudaDeviceSynchronize();  CUDA_CHECK;
    //cpu deallocs
    free(imgIn);
    delete[] imgOut;
#ifdef CONVOLUTION_CONSTANT
    delete[] k;
    delete[] kt;
    // gpu vars deallocs
    cout << "nbytesI: " << nbytesI << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
#endif
#ifdef FEATURE_DETECTION
// d_gX, *d_gY, *d_m11, *d_m12, *d_m22, *d_imgOut11, *d_imgOut12, *d_imgOut22
    delete[] imgOut2;

    delete[] k;
    cout << "nbytesI: " << nbytesI << " nbytesG: " << nbytesG << " nbytesO: " << nbytesO << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_gX); CUDA_CHECK;
    cudaFree(d_gY); CUDA_CHECK;
    cudaFree(d_m11); CUDA_CHECK;
    cudaFree(d_m12); CUDA_CHECK;
    cudaFree(d_m22); CUDA_CHECK;
    cudaFree(d_imgOut11); CUDA_CHECK;
    cudaFree(d_imgOut12); CUDA_CHECK;
    cudaFree(d_imgOut22); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
#endif
#ifdef ISOTROPIC_DIFFUSION
    cout << "nbytesI: " << nbytesI << endl;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_gX); CUDA_CHECK;
    cudaFree(d_gY); CUDA_CHECK;
    delete[] k;
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



