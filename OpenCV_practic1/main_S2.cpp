#include "stdafx.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "harrisDetector.h"

using namespace std;


static string path = "../Udemy/OpenCV 3 - Advanced Image Detection and Reconstruction/Code/images/";


// L25: Harris Corners Detection
int L25(){

//------// Harris:

	// Read input image
	cv::Mat image = cv::imread(path+"church01.jpg", 0);
	if (!image.data)
		return 0;

	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// Display the image
	cv::namedWindow("Original");
	cv::imshow("Original", image);

	// Detect Harris corners
	cv::Mat cornerStrength;
	cv::cornerHarris(image, cornerStrength,
		3,     // neighborhood size
		3,     // aperture size
		0.01); // Harris parameter

	// threshold the corner strengths
	cv::Mat harrisCorners;
	double threshold = 0.0001;
	cv::threshold(cornerStrength, harrisCorners,
		threshold, 255, cv::THRESH_BINARY_INV);

	// Display the corners
	cv::namedWindow("Harris");
	cv::imshow("Harris", harrisCorners);

	// Create Harris detector instance
	HarrisDetector harris;
	// Compute Harris values
	harris.detect(image);
	// Detect Harris corners
	std::vector<cv::Point> pts;
	harris.getCorners(pts, 0.02);
	// Draw Harris corners
	harris.drawOnImage(image, pts);

	// Display the corners
	cv::namedWindow("Corners");
	cv::imshow("Corners", image);


//---------// GFTT:

	


	cv::waitKey(0);
	return 0;
}


int main()
{
	L25();

	return 0;
}