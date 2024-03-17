/////////////////////////////////////////////////////////////////////////////////////////////
///////                                                                               ///////
///////                         Zahra Habibi                                          ///////
///////      Color Transformation (BGR to Rect. 709) using GPU(CUDA) & CPU            ///////
///////                          2024-03-11                                           ///////
///////                                                                               ///////
/////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream> 
#include <future> 
#include "ColorConvertor.h"
#include "MediaHandler.h"

int main() 
{
    // media handler reads the video file and retrieve the images frames
    MediaHandler mediaHandler( "D:/CodeChallenges/Siemens/Cpp_Cuda/MediaFiles/VideoTest.mp4" );

    // color convertor receives the image frame and convert the color space from BGR to Rect. 709
    // using both CPU and GPU
    ColorConvertor colorConvertor(mediaHandler.GetFrameWidth(), 
                                  mediaHandler.GetFrameHeight());
    // diffImage: used for verification of GPU result by looking into the different between CPU and GPU
    cv::Mat diffImage;
    //originImg: image frame from video file
    cv::Mat originImg;
    // looping over the video file and converting colorspace of each frame
    while( mediaHandler.GetFrame( originImg ) ) {

        // launching a thread which converts the image color space on CPU
        std::future<cv::Mat> BT709ColorFrameCPUFuture = std::async(std::launch::async,
            [&]{
                cv::Mat result = colorConvertor.TransformBGR2BT709OnCPU(originImg);
                return result;
            });

        // launching a thread which converts the image color space on GPU
        std::future<cv::Mat> BT709ColorFrameGPUFuture = std::async(std::launch::async,
            [&] {
                cv::Mat result = colorConvertor.TransformBGR2BT709OnGPU(originImg);
                return result;
            });

        // retrieve the CPU results from thread
        cv::Mat BT709ColorFrameCPU = BT709ColorFrameCPUFuture.get();
        // retrieve the GPU results from thread
        cv::Mat BT709ColorFrameGPU = BT709ColorFrameGPUFuture.get();

        // display the video frame image in BGR
        cv::imshow("origin image", originImg);
        // display the converted frame image (Rec. 709) on CPU
        cv::imshow("BT709 Color Frame on CPU", BT709ColorFrameCPU);
        // display the converted frame image (Rec. 709) on GPU
        cv::imshow("BT709 Color Frame on GPU", BT709ColorFrameGPU);

        // Display the difference between CPU & GPU
        // ideally if there is no error in cuda computation, this should a black image
        cv::absdiff(BT709ColorFrameGPU, BT709ColorFrameCPU, diffImage);
        cv::imshow("difference CPU/GPU", diffImage);
        cv::waitKey(30);
    }

    return 0;
}