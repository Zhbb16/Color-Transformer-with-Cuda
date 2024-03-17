#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>

class MediaHandler
{
public:

	MediaHandler( std::string sourceMediaFilePath );

	~MediaHandler();

	bool GetFrame( cv::Mat& frame );

	int GetFrameWidth();

	int GetFrameHeight();

private:

	int _frameWidth;
	int _frameHeight;
	cv::VideoCapture _cap;
};