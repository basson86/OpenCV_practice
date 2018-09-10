#ifndef COLORDETECTOR
#define COLORDETECTOR

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>


class ColorDetector {

private:

	int maxDist;
	
	cv::Vec3b target;

	cv::Mat converted;

	bool useLab;

	cv::Mat result;


public:

	ColorDetector() : maxDist(100), target(0, 0, 0), useLab(false) {};

	ColorDetector(bool lab) : maxDist(100), target(0, 0, 0), useLab(lab) {};

	ColorDetector(uchar blue, uchar green, uchar red, int maxDist, bool lab) : maxDist(maxDist),  useLab(lab) 
	{
		setTargetColor(blue, green, red);
	};

	// Setter & Getter

	// Computes the distance from target color.
	int getDistanceToTargetColor(const cv::Vec3b& color) const {
		return getColorDistance(color, target);
	}

	// Computes the city-block distance between two colors.
	int getColorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) const {

		return abs(color1[0] - color2[0]) +
			abs(color1[1] - color2[1]) +
			abs(color1[2] - color2[2]);

		// Or:
		// return static_cast<int>(cv::norm<int,3>(cv::Vec3i(color[0]-color2[0],color[1]-color2[1],color[2]-color2[2])));

		// Or:
		// cv::Vec3b dist;
		// cv::absdiff(color,color2,dist);
		// return cv::sum(dist)[0];
	}
	// Processes the image. Returns a 1-channel binary image.
	cv::Mat process(const cv::Mat &image);

	cv::Mat operator()(const cv::Mat &image) {

		cv::Mat input;

		if (useLab) { // Lab conversion
			cv::cvtColor(image, input, CV_BGR2Lab);
		}
		else {
			input = image;
		}

		cv::Mat output;
		// compute absolute difference with target color
		cv::absdiff(input, cv::Scalar(target), output);
		// split the channels into 3 images
		std::vector<cv::Mat> images;
		cv::split(output, images);
		// add the 3 channels (saturation might occurs here)
		output = images[0] + images[1] + images[2];
		// apply threshold
		cv::threshold(output,  // input image
			output,  // output image
			maxDist, // threshold (must be < 256)
			255,     // max value
			cv::THRESH_BINARY_INV); // thresholding type

		return output;
	}



	void setTargetColor(uchar blue, uchar green, uchar red)
	{
		target = cv::Vec3b(blue, green, red);

		if (useLab)
		{
			// create a temporary "one-pixel" image
			cv::Mat tmp(1, 1, CV_8UC3);
			tmp.at<cv::Vec3b>(0, 0) = cv::Vec3b(blue, green, red);

			// Converting the target to Lab color space 
			cv::cvtColor(tmp, tmp, CV_BGR2Lab);

			target = tmp.at<cv::Vec3b>(0,0);

		}

	};

	// Sets the color to be detected
	// given in BGR color space
	void setTargetColor(cv::Vec3b color)
	{
		target = color;
	};

	cv::Vec3b getTargetColor() const
	{
		return target;
	};

};


























#endif