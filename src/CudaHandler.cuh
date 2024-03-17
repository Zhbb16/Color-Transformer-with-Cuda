#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct CudaHandlerDevice
{
	CudaHandlerDevice(int frameWidth, int frameHeight) :
		_frameWidth(frameWidth),
		_frameHeight(frameHeight)
	{
	}
	__device__ void TransformBGR2BT709(const uchar* inputBGR, uchar* outputBT709);
	__device__ void Blurring(const uchar* inputBGR, uchar* outputBT709);
	int _frameWidth;
	int _frameHeight;
};

class CudaHandler
{
public:

	CudaHandler(int frameWidth, int frameHeight);

	~CudaHandler();

	void LaunchCudaKernel(const cv::Mat& inputBGR, cv::Mat& outputBT709);

private:
	CudaHandlerDevice _cudaHandlerDevice;
};

__global__ void KernelEntryPoint( CudaHandlerDevice cudaHandlerDevice, const uchar* inputBGR,
	uchar* outputBT709);

