#include "stdafx.h"
#include <iostream>
#include <random>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include <vector>
#include "colordetector.h"
#include "histogram.h"
#include "colorhistogram.h"
#include "contentFinder.h"
#include "imageComparator.h"
#include "integral.h"


using namespace std;


//static string image_path_S1 = "../../Udemy/OpenCV 3 - Getting started with Image processing/Code/Section 1/Images/";
static string image_path_S2 = "../Udemy/OpenCV 3 - Getting started with Image processing/Code/Section 2/Images/";
static string image_path_S3 = "../Udemy/OpenCV 3 - Getting started with Image processing/Code/Section 3/Images/";
static string image_path_S4 = "../Udemy/OpenCV 3 - Getting started with Image processing/Code/Section 4/Images/";



// L13 : Comparing the colors using strategy design pattern
int Excercise_1_13()
{
	// 1. Create image processor object
	ColorDetector cdetect;

	// 2. Read input image
	cv::Mat image = cv::imread(image_path_S2+"boldt.jpg");
	if (image.empty())
		return 0;
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", image);

	// 3. Set input parameters
	cdetect.setTargetColor(230, 190, 130); // here blue sky

	// 4. Process the image and display the result
	cv::namedWindow("result");
	cv::Mat result = cdetect.process(image);
	cv::imshow("result", result);

	// or using functor
	// here distance is measured with the Lab color space
	ColorDetector colordetector(230, 190, 130,  // color
		45, true); // Lab threshold
	cv::namedWindow("result (functor)");
	result = colordetector(image);
	cv::imshow("result (functor)", result);

	// testing floodfill
	cv::floodFill(image,            // input/ouput image
		cv::Point(100, 50),         // seed point
		cv::Scalar(255, 0, 0),  // repainted color
		(cv::Rect*)0,  // bounding rectangle of the repainted pixel set
		cv::Scalar(35, 35, 35),     // low and high difference threshold
		cv::Scalar(35, 35, 35),     // most of the time will be identical
		cv::FLOODFILL_FIXED_RANGE); // pixels are compared to seed color

	cv::namedWindow("Flood Fill result");
	result = colordetector(image);
	cv::imshow("Flood Fill result", image);

	// Creating artificial images to demonstrate color space properties
	cv::Mat colors(100, 300, CV_8UC3, cv::Scalar(100, 200, 150));
	cv::Mat range = colors.colRange(0, 100);
	range = range + cv::Scalar(10, 10, 10);
	range = colors.colRange(200, 300);
	range = range + cv::Scalar(-10, -10, 10);

	cv::namedWindow("3 colors");
	cv::imshow("3 colors", colors);

	cv::Mat labImage(100, 300, CV_8UC3, cv::Scalar(100, 200, 150));
	cv::cvtColor(labImage, labImage, CV_BGR2Lab);
	range = colors.colRange(0, 100);
	range = range + cv::Scalar(10, 10, 10);
	range = colors.colRange(200, 300);
	range = range + cv::Scalar(-10, -10, 10);
	cv::cvtColor(labImage, labImage, CV_Lab2BGR);

	cv::namedWindow("3 colors (Lab)");
	cv::imshow("3 colors (Lab)", colors);

	// brightness versus luminance
	cv::Mat grayLevels(100, 256, CV_8UC3);
	for (int i = 0; i < 256; i++) {
		grayLevels.col(i) = cv::Scalar(i, i, i);
	}

	range = grayLevels.rowRange(50, 100);
	cv::Mat channels[3];
	cv::split(range, channels);
	channels[1] = 128;
	channels[2] = 128;
	cv::merge(channels, 3, range);
	cv::cvtColor(range, range, CV_Lab2BGR);


	cv::namedWindow("Luminance vs Brightness");
	cv::imshow("Luminance vs Brightness", grayLevels);

	cv::waitKey();

	return 0;

}


// Lecture 1-14, practice grab-cut algorithm to seperate foreground from background
int Excercise_1_14()
{
	cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg");

	if (!image.data)
		return 0;

	// display image
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", image);

	// define bounding rectagle
	cv::Rect rectangle(50, 25, 210, 180);

	// the models (internally used)
	cv::Mat bgModel, fgModel;

	// segmentation result
	cv::Mat result;

	//Grab cut segmentation

	cv::grabCut(image,
		result, // segmentation results (foreground)
		rectangle, // rectangle containing foreground 
		bgModel, fgModel,
		5, // number of iterations
		cv::GC_INIT_WITH_RECT); // use rectagle for initialization

	// Get the pixel marked as likely foreground as a "foreground mask" 
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	// or
	// result = result&1 (convert all non-zero pixels to mask)

	// create the canvas
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	
	// copy only the foreground from image to canvas
	image.copyTo(foreground, result);

	//draw the rectangle on the original image
	cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);
	cv::namedWindow("Image with rectangle");
	cv::imshow("Image with rectangle", image);

	//display result
	cv::namedWindow("Foreground object");
	cv::imshow("Foreground object", foreground);

	cv::waitKey();



	return 0;



}

// L16: hue, saturation (HSV color space)
void detectHScolor(const cv::Mat& image,
	double minHue, double maxHue, // Hue interval
	double minSat, double maxSat, // Saturation interval
	cv::Mat& mask) // outputmask
{
	//convert into HSV space
	cv::Mat hsv;
	cv::cvtColor(image, hsv, CV_BGR2HSV);
	// channel[0] is hue
	// channel[1] is saturation
	// channel[2] is value

	// splite the 3 channels into 3 images
	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);

	//Hue masking
	cv::Mat mask1; // below maxHue
	cv::threshold(channels[0], mask1, maxHue, 255, cv::THRESH_BINARY_INV);
	cv::Mat mask2; // above minHue
	cv::threshold(channels[0], mask2, minHue, 255, cv::THRESH_BINARY);

	cv::Mat hueMask; // hue mask
	if (minHue < maxHue)
		hueMask = mask1 & mask2;
	else //if interval croses the zero-degree axis
		hueMask = mask1 | mask2;


	//Saturation masking
	//below maxSat
	cv::threshold(channels[1], mask1, maxSat, 255, cv::THRESH_BINARY_INV);
	//over minSat
	cv::threshold(channels[1], mask1, minSat, 255, cv::THRESH_BINARY);


	cv::Mat satMask;// saturation mask
	satMask = mask1 & mask2;


	//combine mask
	mask = hueMask & satMask;




}



int Excercise_1_16()
{

	// Testing skin detection using HueSaturation Detector

	// read the image
	cv::Mat image = cv::imread(image_path_S3 + "girl.jpg");
	if (!image.data)
		return 0;

	// show original image
	cv::namedWindow("Original image");  
	cv::imshow("Original image", image);

	// detect skin tone
	cv::Mat mask;
	detectHScolor(image,
		160, 10, // hue from 320 degrees to 20 degrees 
		25, 166, // saturation from ~0.1 to 0.65
		mask);

	// show masked image
	cv::Mat detected(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	image.copyTo(detected, mask);
	cv::imshow("Detection result", detected);

	cv::waitKey(0);
	return 0;
}


int Excercise_1_17()
{
	// Read input image
	cv::Mat image = cv::imread(image_path_S4+"group.jpg", 0);
	if (!image.data)
		return 0;

	// save grayscale image
	cv::imwrite(image_path_S4+"groupBW.jpg", image);

	// Display the image
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	// The histogram object
	Histogram1D h;

	// Compute the histogram
	cv::Mat histo = h.getHistogram(image);

	// Loop over each bin
	for (int i = 0; i<256; i++)
		cout << "Value " << i << " = " << histo.at<float>(i) << endl;

	// Display a histogram as an image
	cv::namedWindow("Histogram");
	cv::imshow("Histogram", h.getHistogramImage(image));


	// re-display the histagram with chosen threshold indicated
	cv::Mat hi = h.getHistogramImage(image);
	cv::line(hi, cv::Point(70, 0), cv::Point(70, 255), cv::Scalar(128));
	cv::namedWindow("Histogram with threshold value");
	cv::imshow("Histogram with threshold value", hi);

	cv::Mat thresholded;
	cv::threshold(image, thresholded, 70, 255, cv::THRESH_BINARY);


	cv::namedWindow("Binary image");
	cv::imshow("Binary Image", thresholded);


	cv::waitKey(0);
	return 0;
}

// L-18, apply lookup table to modify image
void colorReduce(cv::Mat& image, int div = 64){

	cv::Mat lookup(1, 256, CV_8U);

	for (int i = 0; i < 256; i++){
		lookup.at<uchar>(i) = i / div*div + div / 2;

		cv::LUT(image, lookup, image);
	}


}

int Excercise_1_18()
{
	// Read input image
	cv::Mat image = cv::imread(image_path_S4 + "group.jpg", 0);
	if (!image.data)
		return 0;

	// The histogram object
	Histogram1D h;

	// Compute the histogram
	cv::Mat histo = h.getHistogram(image);

	//Create an image inversion table
	cv::Mat lut(1, 256, CV_8U); // 1x256 matrix

	for (int i = 0; i < 256; i++)
	{
		// 0 becomes 255, 1 becomes 254, etc.
		lut.at<uchar>(i) = 255 - i;
	}

	//Apply lookup and display negative image
	cv::namedWindow("Negative image");
	cv::imshow("Negative image", h.applyLookUp(image, lut));


	// Stretch the image , setting the 1% of pixles at black and 1% at white to cuttoff
	cv::Mat str = h.stretch(image, 0.01f);


	//Show the result
	cv::namedWindow("Stretched Image");
	cv::imshow("Stretched Image", str);


	//Show the new histogram
	cv::namedWindow("Stretched H");
	cv::imshow("Stretched H", h.getHistogramImage(str));

	cv::waitKey(0);
	return 0;

}

//L-19: histogram equlization
int Excercise_1_19(){

	cv::Mat image = cv::imread(image_path_S4 + "group.jpg", 0);
	if (!image.data)
		return 0;

	Histogram1D h;

	// Equalize the image
	cv::Mat eq = h.equalize(image);

	// Show the result
	cv::namedWindow("Equalized Image");
	cv::imshow("Equalized Image", eq);

	// Show the new histogram
	cv::namedWindow("Equalized H");
	cv::imshow("Equalized H", h.getHistogramImage(eq));


	cv::waitKey(0);
	return 0;
}


// L-21 use Mean Shift Algorithm to Find an object
int Excercise_1_21()
{
	// Read the reference image
	cv::Mat image = cv::imread(image_path_S4 + "baboon01.jpg");

	if (!image.data)
		return 0;

	//initial window position to extract 
	cv::Rect rect(110, 45, 35, 45);
	cv::rectangle(image, rect, cv::Scalar(0, 0, 255));

	// Baboon's face ROI
	cv::Mat imageROI = image(rect);

	cv::namedWindow("Image 1");
	cv::imshow("Image 1", image);

	// Get the Hue histogram of the Baboon's face
	int minSat = 65; // use minSat for thresholding (only consider the pixel with saturation > 65)
	ColorHistogram hc;

	cv::Mat colorhist = hc.getHueHistogram(imageROI, minSat);

	ContentFinder finder;
	finder.setHistogram(colorhist);
	finder.setThreshold(0.2f);

	// Convert to HSV space(just for display)
	cv::Mat hsv;
	cv::cvtColor(image, hsv, CV_BGR2HSV);

	//Split the image
	vector<cv::Mat> v;
	cv::split(hsv, v);

	//Eliminate pixels with low saturation
	cv::threshold(v[1], v[1], minSat, 255, cv::THRESH_BINARY);
	cv::namedWindow("Saturation mask");
	cv::imshow("Saturation mask", v[1]);

	//------
	// Second image
	image = cv::imread(image_path_S4 + "baboon02.jpg");

	cv::namedWindow("Image 2");
	cv::imshow("Image 2", image);

	// Convert to HSV space
	cv::cvtColor(image, hsv, CV_BGR2HSV);

	//Get back-projection of hue histogram --> the probability map according to given model histogram 
	int ch[1] = { 0 };
	finder.setThreshold(-1.0f); // no thresholding
	cv::Mat result = finder.find(hsv, 0.0f, 180.0f, ch);

	// Display back projection result
	cv::namedWindow("Backprojection on second image");
	cv::imshow("Backprojection on second image", result);


	// initial window position
	cv::rectangle(image, rect, cv::Scalar(0, 0, 255));

	// search object with mean shift
	cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
		10, // iterate max 10 times
		1); // or until the change in centroid position is less than 1px

	// put the probability map of the whole image : "result" to meanShift function
	std::cout << "meanshift=" << cv::meanShift(result, rect, criteria) << endl;

	// draw output window
	cv::rectangle(image, rect, cv::Scalar(0, 255, 0));

	// Display image
	cv::namedWindow("Image 2 result");
	cv::imshow("Image 2 result", image);

	cv::waitKey(0);
	return 0;
}

//Retrieving similar image content
int Excercise_1_22()
{
	// Read reference image
	cv::Mat image = cv::imread(image_path_S4 + "waves.jpg");
	if (!image.data)
		return 0;

	// Display image
	cv::namedWindow("Query Image");
	cv::imshow("Query Image", image);


	ImageComparator c;
	c.setReferenceImage(image);

	//Read an image and compare it with reference
	cv::Mat input = cv::imread(image_path_S4 + "dog.jpg");
	cout << "wave vs dog: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "marais.jpg");
	cout << "waves vs marais: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "bear.jpg");
	cout << "waves vs bear: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "beach.jpg");
	cout << "waves vs beach: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "polar.jpg");
	cout << "waves vs polar: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "moose.jpg");
	cout << "waves vs moose: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "lake.jpg");
	cout << "waves vs lake: " << c.compare(input) << endl;

	// Read an image and compare it with reference
	input = cv::imread(image_path_S4 + "fundy.jpg");
	cout << "waves vs fundy: " << c.compare(input) << endl;

	cv::waitKey(0);
	return 0;
}



// L-23-1: integral image and adaptive threshold
int Excercise_1_23_1(){

	cv::Mat image = cv::imread(image_path_S4+"book.png",0);
	if (!image.data)
		return 0;
	// rotate the image for easier display
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// display original image
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", image);

	// using a fixed threshold 
	cv::Mat binaryFixed;
	cv::Mat binaryAdaptive;
	cv::threshold(image, binaryFixed, 70, 255, cv::THRESH_BINARY);

	// using as adaptive threshold
	int blockSize = 21; // size of the neighborhood
	int threshold = 10;  // pixel will be compared to (mean-threshold)

	int64 time;
	time = cv::getTickCount();
	cv::adaptiveThreshold(image,           // input image
						  binaryAdaptive,  // output binary image
						  255,             // max value for output
						  cv::ADAPTIVE_THRESH_MEAN_C, // adaptive method
						  cv::THRESH_BINARY, // threshold type
						  blockSize,       // size of the block
						  threshold);      // threshold used
	time = cv::getTickCount() - time;
	std::cout << "time (adaptiveThreshold)= " << time << std::endl;

	// compute integral image
	IntegralImage<int, 1> integral(image);

	// test integral result
	std::cout << "sum=" << integral(18, 45, 30, 50) << std::endl;
	cv::Mat test(image, cv::Rect(18, 45, 30, 50));
	cv::Scalar t = cv::sum(test);
	std::cout << "sum test=" << t[0] << std::endl;

	cv::namedWindow("Fixed Threshold");
	cv::imshow("Fixed Threshold", binaryFixed);

	cv::namedWindow("Adaptive Threshold");
	cv::imshow("Adaptive Threshold", binaryAdaptive);

	cv::Mat binary = image.clone();

	time = cv::getTickCount();
	int nl = binary.rows; // number of lines
	int nc = binary.cols; // total number of elements per line

	// compute integral image
	cv::Mat iimage;
	cv::integral(image, iimage, CV_32S);

	// for each row
	int halfSize = blockSize / 2;
	for (int j = halfSize; j<nl - halfSize - 1; j++) {

		// get the address of row j
		uchar* data = binary.ptr<uchar>(j);
		int* idata1 = iimage.ptr<int>(j - halfSize);
		int* idata2 = iimage.ptr<int>(j + halfSize + 1);

		// for pixel of a line
		for (int i = halfSize; i<nc - halfSize - 1; i++) {

			// compute sum
			int sum = (idata2[i + halfSize + 1] - idata2[i - halfSize] -
				idata1[i + halfSize + 1] + idata1[i - halfSize]) / (blockSize*blockSize);

			// apply adaptive threshold
			if (data[i]<(sum - threshold))
				data[i] = 0;
			else
				data[i] = 255;
		}
	}

	// add white border
	for (int j = 0; j<halfSize; j++) {
		uchar* data = binary.ptr<uchar>(j);

		for (int i = 0; i<binary.cols; i++) {
			data[i] = 255;
		}
	}
	for (int j = binary.rows - halfSize - 1; j<binary.rows; j++) {
		uchar* data = binary.ptr<uchar>(j);

		for (int i = 0; i<binary.cols; i++) {
			data[i] = 255;
		}
	}
	for (int j = halfSize; j<nl - halfSize - 1; j++) {
		uchar* data = binary.ptr<uchar>(j);

		for (int i = 0; i<halfSize; i++) {
			data[i] = 255;
		}
		for (int i = binary.cols - halfSize - 1; i<binary.cols; i++) {
			data[i] = 255;
		}
	}

	time = cv::getTickCount() - time;
	std::cout << "time integral= " << time << std::endl;

	cv::namedWindow("Adaptive Threshold (integral)");
	cv::imshow("Adaptive Threshold (integral)", binary);

	// adaptive threshold using image operators
	time = cv::getTickCount();
	cv::Mat filtered;
	cv::Mat binaryFiltered;
	// box filter compute avg of pixels over a rectangular region
	cv::boxFilter(image, filtered, CV_8U, cv::Size(blockSize, blockSize));
	// check if pixel greater than (mean + threshold)
	binaryFiltered = image >= (filtered - threshold);
	time = cv::getTickCount() - time;

	std::cout << "time filtered= " << time << std::endl;

	cv::namedWindow("Adaptive Threshold (filtered)");
	cv::imshow("Adaptive Threshold (filtered)", binaryFiltered);

	cv::waitKey(0);
	return 0;

}

// L-23-2: integral image and tracking
int Excercise_1_23_2(){

	// Open image
	cv::Mat image = cv::imread(image_path_S4+"bike55.bmp", 0);
	// define image roi
	int xo = 97, yo = 112;
	int width = 25, height = 30;
	cv::Mat roi(image, cv::Rect(xo, yo, width, height));

	// compute sum
	// returns a Scalar to work with multi-channel images
	cv::Scalar sum = cv::sum(roi);
	std::cout << sum[0] << std::endl;

	// compute integral image
	cv::Mat integralImage;
	cv::integral(image, integralImage, CV_32S);
	// get sum over an area using three additions/subtractions
	int sumInt = integralImage.at<int>(yo + height, xo + width)
		- integralImage.at<int>(yo + height, xo)
		- integralImage.at<int>(yo, xo + width)
		+ integralImage.at<int>(yo, xo);
	std::cout << sumInt << std::endl;

	// histogram of 16 bins
	Histogram1D h;
	h.setNBins(16);
	// compute histogram over image roi 
	cv::Mat refHistogram = h.getHistogram(roi);

	cv::namedWindow("Reference Histogram");
	cv::imshow("Reference Histogram", h.getHistogramImage(roi, 16));
	std::cout << refHistogram << std::endl;

	// first create 16-plane binary image
	cv::Mat planes;
	convertToBinaryPlanes(image, planes, 16);
	// then compute integral image
	IntegralImage<float, 16> intHisto(planes);


	// for testing compute a histogram of 16 bins with integral image
	cv::Vec<float, 16> histogram = intHisto(xo, yo, width, height);
	std::cout << histogram << std::endl;

	cv::namedWindow("Reference Histogram (2)");
	cv::Mat im = h.getImageOfHistogram(cv::Mat(histogram), 16);
	cv::imshow("Reference Histogram (2)", im);

	// search in second image
	cv::Mat secondImage = cv::imread(image_path_S4 + "bike65.bmp", 0);
	if (!secondImage.data)
		return 0;

	// first create 16-plane binary image
	convertToBinaryPlanes(secondImage, planes, 16);
	// then compute integral image
	IntegralImage<float, 16> intHistogram(planes);

	// compute histogram of 16 bins with integral image (testing)
	histogram = intHistogram(135, 114, width, height);
	std::cout << histogram << std::endl;

	cv::namedWindow("Current Histogram");
	cv::Mat im2 = h.getImageOfHistogram(cv::Mat(histogram), 16);
	cv::imshow("Current Histogram", im2);

	std::cout << "Distance= " << cv::compareHist(refHistogram, histogram, CV_COMP_CORREL) << std::endl;

	double maxSimilarity = 0.0;
	int xbest, ybest;
	// loop over a horizontal strip around girl location in initial image
	for (int y = 110; y<120; y++) {
		for (int x = 0; x<secondImage.cols - width; x++) {


			// compute histogram of 16 bins using integral image
			histogram = intHistogram(x, y, width, height);
			// compute distance with reference histogram
			double distance = cv::compareHist(refHistogram, histogram, CV_COMP_CORREL);
			// find position of most similar histogram
			if (distance>maxSimilarity) {

				xbest = x;
				ybest = y;
				maxSimilarity = distance;
			}

			std::cout << "Distance(" << x << "," << y << ")=" << distance << std::endl;
		}
	}

	std::cout << "Best solution= (" << xbest << "," << ybest << ")=" << maxSimilarity << std::endl;

	// draw a rectangle around target object
	cv::rectangle(image, cv::Rect(xo, yo, width, height), 0);
	cv::namedWindow("Initial Image");
	cv::imshow("Initial Image", image);

	cv::namedWindow("New Image");
	cv::imshow("New Image", secondImage);

	// draw rectangle at best location
	cv::rectangle(secondImage, cv::Rect(xbest, ybest, width, height), 0);
	// draw rectangle around search area
	cv::rectangle(secondImage, cv::Rect(0, 110, secondImage.cols, height + 10), 255);
	cv::namedWindow("Object location");
	cv::imshow("Object location", secondImage);

	cv::waitKey();
	return 0;

}

// Main function entry point




int main()
{
	
	return Excercise_1_21();
}