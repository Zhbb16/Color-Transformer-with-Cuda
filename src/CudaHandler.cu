#include "CudaHandler.cuh"

//=============================================================================
//  C O N S T R U C T O R (S) / D E S T R U C T O R      S E C T I O N S   

//=============================================================================
//  
CudaHandler::CudaHandler(int frameWidth, int frameHeight) :
    _cudaHandlerDevice(frameWidth, frameHeight)
{
}

//=============================================================================
//  
CudaHandler::~CudaHandler()
{
}

//=============================================================================
//  M T H O D S     S E C T I O N S   

//=============================================================================
//  
__device__ void 
CudaHandlerDevice::TransformBGR2BT709(const uchar* inputBGR, uchar* outputBT709)
{
    // x,y: are the global thread indices in 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure gpu thread indices are in the range of image sizes
    if (x < _frameWidth && y < _frameHeight) {

        // map the global thread index to the image pixel
        int idx = (y * _frameWidth + x) * 3;

        // retrieve the image pixel values Blue, Green, Red
        float b = inputBGR[idx] / 255.0f;
        float g = inputBGR[idx + 1] / 255.0f;
        float r = inputBGR[idx + 2] / 255.0f;

        // apply gamma correction, note that this function is expensive on cuda
        b = powf(b, 2.2f);
        g = powf(g, 2.2f);
        r = powf(r, 2.2f);;

        // Linear transformation to BT.709 RGB
        float bt709_r = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
        float bt709_g = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
        float bt709_b = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

        // Non-linear transformation
        bt709_r = (bt709_r <= 0.018f) ? (bt709_r / 4.5f) : powf((bt709_r + 0.099f) / 1.099f, 1.0f / 0.45f);
        bt709_g = (bt709_g <= 0.018f) ? (bt709_g / 4.5f) : powf((bt709_g + 0.099f) / 1.099f, 1.0f / 0.45f);
        bt709_b = (bt709_b <= 0.018f) ? (bt709_b / 4.5f) : powf((bt709_b + 0.099f) / 1.099f, 1.0f / 0.45f);

        // clamp the values between 0 and 1
        if (bt709_r > 1.f) {
            bt709_r = 1.f;
        }
        if (bt709_r < 0.f) {
            bt709_r = 0.f;
        }
        if (bt709_g > 1.f) {
            bt709_g = 1.f;
        }
        if (bt709_g < 0.f) {
            bt709_g = 0.f;
        }
        if (bt709_b > 1.f) {
            bt709_b = 1.f;
        }
        if (bt709_b < 0.f) {
            bt709_b = 0.f;
        }

        // Scale to 8-bit range
        outputBT709[idx] = (uchar)(bt709_b * 255.f);
        outputBT709[idx + 1] = (uchar)(bt709_g * 255.f);
        outputBT709[idx + 2] = (uchar)(bt709_r * 255.f);
    }
}

//=============================================================================
//  
__global__ void 
KernelEntryPoint( CudaHandlerDevice cudaHandlerDevice, const uchar* inputBGR, 
    uchar* outputBT709 )
{
    cudaHandlerDevice.TransformBGR2BT709(inputBGR, outputBT709);
}

//=============================================================================
//  
void 
CudaHandler::LaunchCudaKernel( const cv::Mat& inputBGR, cv::Mat& outputBT709)
{
    // allocate device pointers
    uchar* deviceInputBGR;
    uchar* deviceOutputBT709;

    cudaMalloc(&deviceInputBGR, inputBGR.rows * inputBGR.cols * inputBGR.channels() * sizeof(uchar));
    cudaMalloc(&deviceOutputBT709, inputBGR.rows * inputBGR.cols * inputBGR.channels() * sizeof(uchar));

    // copy from host to device
    cudaMemcpy(deviceInputBGR, inputBGR.ptr<uchar>(0),
        inputBGR.rows * inputBGR.cols * inputBGR.channels() * sizeof(uchar), cudaMemcpyHostToDevice);

    // call CUDA kernel
    // a 2D block of threads 
    dim3 block(32, 32, 1);
    int width = inputBGR.cols;
    int height = inputBGR.rows;
    // a 2D grid of blocks for mapping the input image pixels to the threads
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    // instantiate the cuda handler for passing to cuda kernel
    CudaHandlerDevice cudaHandlerDevice(width, height);

    // launching cuda kernel
    KernelEntryPoint <<<grid, block >>> (cudaHandlerDevice, deviceInputBGR, deviceOutputBT709);

    // wait for the cude kernel to finish the computation
    cudaDeviceSynchronize();

    // copy back the result from gpu to host
    outputBT709.create(inputBGR.rows, inputBGR.cols, inputBGR.type());
    cudaMemcpy(outputBT709.data, deviceOutputBT709,
        outputBT709.rows * outputBT709.cols * outputBT709.channels() * sizeof(uchar),
        cudaMemcpyDeviceToHost);

    // free the cuda memories
    cudaFree(deviceInputBGR);
    cudaFree(deviceOutputBT709);
}