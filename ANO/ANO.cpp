// ANO.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"

#define M_PI 3.14159265358979323846
using namespace cv;
using namespace std;

void Recursion(Mat &image, Mat &indexedImage, Mat &coloredImage, int y, int x, int index, Vec3b color) {
	if (x > image.cols || x < 0)
		return;
	if (y > image.rows || y < 0)
		return;

	if (image.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
	{
		indexedImage.at<float>(y, x) = index;
		coloredImage.at<Vec3b>(y, x) = color;
		Recursion(image, indexedImage, coloredImage, y, x + 1, index, color);
		Recursion(image, indexedImage, coloredImage, y + 1, x, index, color);
		Recursion(image, indexedImage, coloredImage, y, x - 1, index, color);
		Recursion(image, indexedImage, coloredImage, y - 1, x, index, color);
	}
}

void ComputeCenterOfObjects(Mat &indexedImage, Mat &coloredImage, std::list<int> indexes) {
	if (indexes.size() > 0)
	{
		int index = 0;
		std::list<int>::iterator it = indexes.begin();
		std::list<Point> centers;

		std::cout << "Computing centers" << std::endl;
		while (it != indexes.end())
		{
			index = *it;
			float m10 = 0.0;
			float m01 = 0.0;
			float m00 = 0.0;
			for (int y = 0; y < indexedImage.rows; y++) {
				for (int x = 0; x < indexedImage.cols; x++) {
					if (indexedImage.at<float>(y, x) == index) {
						m10 += pow(x, 1) * pow(y, 0) * indexedImage.at<float>(y, x);
						m01 += pow(x, 0) * pow(y, 1) * indexedImage.at<float>(y, x);
						m00 += pow(x, 0) * pow(y, 0) * indexedImage.at<float>(y, x);
					}
				}
			}
			centers.push_back(Point(m10 / m00, m01 / m00));

			it++;
		}
		std::list<Point>::iterator centerIterator = centers.begin();

		for (centerIterator = centers.begin(); centerIterator != centers.end(); centerIterator++)
		{
			int x = centerIterator->x;
			int y = centerIterator->y;
			string centerPoint = to_string(x) + ":" + to_string(y);
			cv::putText(coloredImage,
				centerPoint,
				cv::Point(x, y), // Coordinates
				cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
				0.5, // Scale. 2.0 = 2x bigger
				cv::Scalar(255, 255, 255)); // BGR Color
			//std::cout << centerPoint<< std::endl;

		}
	}
}

void ImageTresholding()
{
	Mat image = imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageGray;
	image.convertTo(imageGray, CV_32FC1, 1.0 / 255.0);
	Mat indexedImage = Mat::zeros(image.size(), CV_32FC1);
	Mat coloredImage = Mat::zeros(image.size(), CV_8UC3);
	std::list<int> indexes;
	int index = rand() % 255 + 1;
	Vec3b color = Vec3b(rand() % 255, rand() % 255, rand() % 255);
	for (int y = 0; y < imageGray.rows; y++) {
		for (int x = 0; x < imageGray.cols; x++) {
			if (imageGray.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
			{
				indexes.push_back(index);
				Recursion(imageGray, indexedImage, coloredImage, y, x, index, color);
				index = rand() % 255 + 1;
				color = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
			}
		}
	}

	ComputeCenterOfObjects(indexedImage, coloredImage, indexes);

	imshow("Initial", imageGray);
	imshow("tresshold", indexedImage);
	imshow("color", coloredImage);


	cv::waitKey(0);


}



int main()
{
	ImageTresholding();
    
}

