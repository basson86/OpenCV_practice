// OpenCV_practic1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <random>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;

static string image_path_S1 = "../Udemy/OpenCV 3 - Getting started with Image processing/Code/Section 1/Images/";
static string image_path_S2 = "../Udemy/OpenCV 3 - Getting started with Image processing/Code/Section 2/Images/";


void onMouse(int event, int x, int y, int flags, void* param){


	cv::Mat *im = reinterpret_cast<cv::Mat*>(param);

	switch (event){


	case cv::EVENT_LBUTTONDOWN:
		std::cout << "at (" << x << "," << y << ") value is: "
			<< static_cast<int> (im->at<uchar>(cv::Point(x, y))) << std::endl;

	break;

	}

}

// Exercie Section 1 , L 3
int Excercise_1_3()
{
	
	cv::Mat image = cv::imread(image_path_S1 + "/puppy.bmp", cv::IMREAD_COLOR);

	cout << "This image is " << image.rows << "x " << image.cols << std::endl;

	if (image.empty())
	{
		return 0;
	}

	cv::namedWindow("Original image");
	cv::imshow("Original image", image);

	cv::setMouseCallback("Original image", onMouse, reinterpret_cast<void*>(&image));


	cv::Mat result;
	cv::flip(image, result, 1);

	cv::namedWindow("Output image");
	cv::imshow("Output image", result);

	cv::waitKey(0);
	return 0;
}

// Exercie Section 1 , L 4
int Excercise_1_4()
{
	const auto function=[](){

		cv::Mat ima(500, 500, CV_8U, 50);
		return ima;
	};



	
	
	
	cv::Mat image1(240, 320, CV_8U, 100);
	image1.create(200, 200, CV_8U);
	image1 = 200;


	//cv::Mat image2(240, 320, CV_8UC3, cv::Scalar(0, 0, 255));
	cv::Mat image2(cv::Size(320, 240), CV_8UC3);
	
	cv::Mat image3 = cv::imread(image_path_S1 + "/puppy.bmp");

	// Shallow copy: image 4 & image 1 shares the same memory as image 3
	cv::Mat image4(image3);
	image1 = image3;

	// Deep copy: create a independent copy of memory: image2 and image 5
	image3.copyTo(image2);
	cv::Mat image5 = image3.clone();


	cv::flip(image3, image3, 1);
	cv::imshow("Image 3", image3);
	cv::imshow("Image 1", image1);
	cv::imshow("Image 2", image2);
	cv::imshow("Image 4", image4);
	cv::imshow("Image 5", image5);
	cv::waitKey(0);


	// Conver to floating point image

	cv::Mat gray = function();

	image1 = cv::imread("puppy.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	image1.convertTo(image2, CV_32F, 1 / 255.0, 0.0);

	cv::imshow("image1 floating point", image2);

	// Matrix operation

	cv::Matx33d matrix(3.0, 2.0, 1.0,
						2.0, 1.0, 3.0,
						1.0, 2.0, 3.0);

	cv::Matx31d vector(5.0, 1.0, 3.0);
	cv::Matx31d result = matrix * vector;

	cv::waitKey(0);


	return 0;
}

// Exercie Section 1 , L 5
int Excercise_1_5()
{
	cv::namedWindow("image");
	cv::Mat image = cv::imread(image_path_S1 + "puppy.bmp");
	cv::Mat logo = cv::imread(image_path_S1 + "smalllogo.png");

	cv::Mat imageROI(image, cv::Rect(image.cols - logo.cols, image.rows - logo.rows, logo.cols, logo.rows));

	// Copy the logo to ROI
	/*logo.copyTo(imageROI);
	cv::imshow("Image", image);
	cv::waitKey(0);*/

	// different ways to create ROI by Rect and Range
	/*imageROI = image(cv::Rect(image.cols - logo.cols, image.rows - logo.rows, logo.cols, logo.rows))
	imageROI = image(cv::Range(image.rows - logo.rows, image.rows), cv::Range(image.cols - logo.cols, image.cols));*/


	// Copy only the white portion to image by using mask
	cv::Mat mask(logo);
	logo.copyTo(imageROI, mask);

	cv::imshow("image", image);
	cv::waitKey(0);

	

	return 0;

}

void salt(cv::Mat image, int n)
{

	std::default_random_engine generator;
	std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
	std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

	int i, j;

	for (int k = 0; k < n; k++)
	{
		i = randomCol(generator);
		j = randomRow(generator);

		if (image.type() == CV_8UC1)
		{
			image.at<uchar>(j, i) = 255;

		}
		else if (image.type() == CV_8UC3)
		{
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;

			


			// alternative way to write the BGR
			image.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 255, 255);


		}
	}


	// image content is easier to index by delaring it as Mat_<type> ( equivalent to Mat1b, Mat1f...)
	cv::Mat_<uchar> img(image);
	img(10, 100) = 0;


}


int Excercise_1_6()
{
	cv::Mat image = cv::imread(image_path_S2+"boldt.jpg", 1);
	salt(image, 3000);


	cv::namedWindow("Image");
	cv::imshow("Image", image);

	cv::waitKey(0);

	return 0;

}


// Excercise for S1-7 : Scanning an image  with Pointers

// Helper function for color reduction
void colorReduce(cv::Mat image, int div = 64)
{
	int nl = image.rows;
	int nc = image.cols*image.channels();

	for (int j = 0; j < nl; j++)
	{
		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++)
		{
			//data[i] = data[i] / div* div + div / 2;
			*data++ = *data / div*div + div / 2;
		}


	}

}


void colorReduceIO(	const cv::Mat &image,
					cv::Mat &result,
					int div = 64)
{
	int nl = image.rows;
	int nc = image.cols;
	int nchannels = image.channels();

	result.create(image.rows, image.cols, image.type());
	for (int j = 0; j < nl; j++)
	{
		const uchar* data_in = image.ptr<uchar>(j);
		uchar* data_out = result.ptr<uchar>(j);

		for (int i = 0; i < nc*nchannels; i++)
		{
			data_out[i] = data_in[i] / div*div + div / 2;
		}
	}
}


void colorReduce6(cv::Mat &image,
				  int div = 64)
{
	int nl = image.rows;
	int nc = image.cols*image.channels();
 
	if (image.isContinuous())
	{
		// treate as one row continuous array
		nc = nl*nc;
		nl = 1;

		// equivalent as :
		image.reshape(1, 1);

	}

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);

	uchar mask = 0xFF << n;
	uchar div2 = div >> 1;

	for (int j = 0; j < nl; j++)
	{
		uchar* data= image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++){
			*data &= mask;
			*data++ += div2;
		}
	}
}


int Excercise_1_7()
{
	cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg");
	cv::Mat imageClone = image.clone();

	colorReduceIO(image, imageClone);

	//colorReduce6(image);

	cv::namedWindow("Image Result");
	cv::imshow("Image Result", imageClone);


	cv::waitKey(0);
	
	return 0;

}

// Lector 1-8, scanning using iterator
void colorReduce10(cv::Mat image, int div=64)
{
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	uchar mask = 0xFF << n;
	uchar div2 = div >> 1;

	
	/*cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();*/
	
	// A simpler way to get iterator without specifying type
	cv::Mat_<cv::Vec3b> cimage = image;
	cv::Mat_<cv::Vec3b>::iterator it = cimage.begin();
	cv::Mat_<cv::Vec3b>::iterator itend = cimage.end();


	for (; it != itend; ++it)
	{
		(*it)[0] &= mask;
		(*it)[0] += div2;
		(*it)[1] &= mask;
		(*it)[1] += div2;
		(*it)[2] &= mask;
		(*it)[2] += div2;

	}

}

int Excercise_1_8()
{
	cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg");
	
	colorReduce10(image);

	cv::namedWindow("Image Result");
	cv::imshow("Image Result", image);

	cv::waitKey(0);

	return 0;

}



// L 1-9 : writing efficient image scanning loops:
void colorReduceFast(cv::Mat image, int div=64)
{
	int nl = image.rows;
	int nc = image.cols*image.channels();
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	
	uchar mask = 0xFF << n;
	int div2 = div >> 1;

	for (int j = 0; j < nl; j++)
	{
		uchar* data = image.ptr<uchar>(j);
		
		for (int i = 0; i < nc; i++)
		{
			*(data+i) &= mask;
			*(data+i) += div2;
		}
	}


}

int Excercise_1_9()
{
	cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg");
	
	const int64 start = cvGetTickCount();
	colorReduceFast(image);
	double duration = (cvGetTickCount() - start) / cvGetTickFrequency();

	cout << "colorReduceFase took " << duration << "ms\n";

	cv::namedWindow("Image Result");
	cv::imshow("Image Result", image);

	cv::waitKey(0);

	return 0;

}
// S1-L10: scanning an image with neighbor access: for both greaylevel or RGB

void sharpen(const cv::Mat &image, cv::Mat &result)
{
	result.create(image.size(), image.type());
	int nchannels = image.channels();

	for (int j = 1; j < image.rows - 1; j++)
	{
		const uchar* previous = image.ptr<const uchar>(j - 1);
		const uchar* current = image.ptr<const uchar>(j);
		const uchar* next = image.ptr<const uchar>(j + 1);
		
		uchar* output = result.ptr<uchar>(j);

		for (int i = nchannels; i < (image.cols - 1)*nchannels; i++)
		{
			*output++ = cv::saturate_cast<uchar>(5 * current[i] - current[i - nchannels] - current[i + nchannels] - previous[i] - next[i]);
		}


	}

	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows - 1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols - 1).setTo(cv::Scalar(0));


}

// using kernel function
void sharpen2D(const cv::Mat& image, cv::Mat& result)
{
	
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));

	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;


	cv::filter2D(image, result, image.depth(), kernel);


}



int Excercise_1_10()
{
	cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg");
	//cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	if (!image.data)
		return 0;

	const int64 start = cvGetTickCount();
	cv::Mat imageC=image.clone();
	sharpen2D(image, imageC);
	double duration = (cvGetTickCount() - start) / cvGetTickFrequency();

	cout << "sharpening image  took " << duration << "ms\n";

	cv::imshow("original image", image);
	cv::imshow("sharpened image", imageC);

	cv::waitKey(0);

	return 0;

}

// L11 : simple image artithmetic
int Excercise_1_11()
{
	cv::Mat image1 = cv::imread(image_path_S2 + "boldt.jpg");
	cv::Mat image2 = cv::imread(image_path_S2 + "rain.jpg");

	if (!image1.data || !image2.data)
		return 0;

	cv::namedWindow("Image 1");
	cv::imshow("Image 1", image1);
	cv::namedWindow("Image 2");
	cv::imshow("Image 2", image2);


	cv::Mat result;

	cv::addWeighted(image1, 0.7, image2, 0.9, 0, result);
	cv::namedWindow("result");
	cv::imshow("result", result);


	result = 0.7*image1 + 0.9*image2;

	cv::namedWindow("result with overloaded operators");
	cv::imshow("result with overloaded operators", result);

	// perform splitting: copy a particular channel of color image

	image2 = cv::imread(image_path_S2 + "rain.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	std::vector<cv::Mat> planes;

	// add rain only on Blue channel
	cv::split(image1, planes);

	planes[0] += image2;
	// merget the 3 channel back
	cv::merge(planes, result);

	cv::namedWindow("result on blue channel");
	cv::imshow("result on blue channel", result);


	cv::waitKey(0);

	return 0;

}
void wave(const cv::Mat & image, cv::Mat &result)
{
	// two map storing x & y after mapping
	cv::Mat srcX(image.rows, image.cols, CV_32F);
	cv::Mat srcY(image.rows, image.cols, CV_32F);


	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{

			// flipping the image
			srcX.at<float>(i, j) = image.cols-j-1;
			srcY.at<float>(i, j) = i;
			
			// create sinusodual effect
			/*srcX.at<float>(i, j) = j;
			srcY.at<float>(i, j) = i + 3 * sin(j / 6.0);*/



		}

	}


	cv::remap(image, result, srcX, srcY, cv::INTER_LINEAR);

}


// L12 : Remapping an image
int Excercise_1_12()
{
	cv::Mat image = cv::imread(image_path_S2 + "boldt.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat result;

	wave(image, result);

	cv::namedWindow("Remapped image");
	cv::imshow("Remapped image", result);
	cv::waitKey(0);

	return 0;
}

// Main function entry point

int main()
{
	
	return Excercise_1_12();
}

