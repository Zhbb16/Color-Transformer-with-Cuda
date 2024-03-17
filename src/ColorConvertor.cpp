#include "ColorConvertor.h"

//=============================================================================
//  C O N S T R U C T O R (S) / D E S T R U C T O R      S E C T I O N S   

//=============================================================================
//  
ColorConvertor::ColorConvertor( int frameWidth, int frameHeight ) :
    _cudaHandler(frameWidth, frameHeight ),
    _cpuHandler()
{
}

//=============================================================================
//  
ColorConvertor::~ColorConvertor() 
{
}

//=============================================================================
//  
CPUHandler::CPUHandler()
{
}

//=============================================================================
//  M T H O D S     S E C T I O N S   

//=============================================================================
//  
cv::Mat
CPUHandler::TransformBGR2Rect709( const cv::Mat& inputBGR ) 
{
    cv::Mat outputBT709 = cv::Mat::zeros(inputBGR.size(), inputBGR.type());;
    cv::Mat inputImg = inputBGR.clone();

    // normalize pixel to 0 and 1
    inputBGR.convertTo(inputImg, CV_32FC3, 1.f / 255.f); 
    cv::pow(inputImg, 2.2f, inputImg); // gamma correction

    // Linear transformation 
    for (int y = 0; y < inputImg.rows; ++y) {
        for (int x = 0; x < inputImg.cols; ++x) {
            cv::Vec3f& pixel = inputImg.at<cv::Vec3f>(y, x);

            float b = pixel[0];  // Blue channel
            float g = pixel[1];  // Green channel
            float r = pixel[2];  // Red channel

            pixel[2] = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b; //r
            pixel[1] = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b; //g
            pixel[0] = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b; //b
        }
    }

    // Non-linear transformation
    for (int i = 0; i < inputImg.rows; ++i) {
        for (int j = 0; j < inputImg.cols; ++j) {
            cv::Vec3f& pixel = inputImg.at<cv::Vec3f>(i, j);
            for (int k = 0; k < 3; ++k) {
                pixel[k] = (pixel[k] <= 0.018f) ? (pixel[k] / 4.5f) : powf((pixel[k] + 0.099f)
                    / 1.099f, 1.0f / 0.45f);
                // clamp the pixel values between 0 and 1
                if (pixel[k] < 0.f) {
                    pixel[k] = 0.f;
                }
                if (pixel[k] > 1.f) {
                    pixel[k] = 1.f;
                }
            }
        }
    }

    // convert back float values to unsigned int8 datatype
    inputImg *= 255.f;
    inputImg.convertTo(outputBT709, CV_8UC3);
    return outputBT709;
}

//=============================================================================
//  
cv::Mat 
ColorConvertor::TransformBGR2BT709OnCPU( const cv::Mat &inputBGR )
{
    cv::Mat outputBT709 = _cpuHandler.TransformBGR2Rect709( inputBGR );
    return outputBT709;
}

//=============================================================================
//  
cv::Mat
ColorConvertor::TransformBGR2BT709OnGPU(const cv::Mat& inputBGR)
{
    cv::Mat outputBT709;
    // launch cuda kernel
    _cudaHandler.LaunchCudaKernel( inputBGR, outputBT709 );
    return outputBT709;
}