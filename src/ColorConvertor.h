#include "opencv2/opencv.hpp"
#include "CudaHandler.cuh"

struct CPUHandler
{
	CPUHandler();
	cv::Mat TransformBGR2Rect709( const cv::Mat& inputBGR );
};

class ColorConvertor
{

	public:
		ColorConvertor( int frameWidth, int frameHeight  );
		~ColorConvertor();
		cv::Mat TransformBGR2BT709OnCPU(const cv::Mat& inputBGR);
		cv::Mat TransformBGR2BT709OnGPU(const cv::Mat& inputBGR);

	private:
		CudaHandler _cudaHandler;
		CPUHandler _cpuHandler;
};