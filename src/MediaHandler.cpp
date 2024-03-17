#include "MediaHandler.h"

//=============================================================================
//  C O N S T R U C T O R (S) / D E S T R U C T O R      S E C T I O N S   

//=============================================================================
//  
MediaHandler::MediaHandler(std::string sourceMediaFilePath) :
	_cap( sourceMediaFilePath )
{

	if (!_cap.isOpened()) {
		std::cerr << "Error: Failed to open video file" << std::endl;
	}
	_frameWidth = int(_cap.get(3));
	_frameHeight = int(_cap.get(4));
}

//=============================================================================
//  
MediaHandler::~MediaHandler() 
{
}

//=============================================================================
//  M T H O D S     S E C T I O N S   

//=============================================================================
//  
bool
MediaHandler::GetFrame( cv::Mat &frame )
{
	// retrieve the frame 
	if (_cap.read(frame)) {
		return true;
	}
	return false;
}

//=============================================================================
//  
int 
MediaHandler::GetFrameWidth() {
	return _frameWidth;
}

//=============================================================================
//  
int
MediaHandler::GetFrameHeight() {
	return _frameHeight;
}
