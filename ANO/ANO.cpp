// ANO.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "ObjectFeature.h"

#define M_PI 3.14159265358979323846
using namespace cv;
using namespace std;

void Recursion(Mat &image, Mat &indexedImage, Mat &coloredImage, int y, int x, FeatureList obj) {
	if (x > image.cols || x < 0)
		return;
	if (y > image.rows || y < 0)
		return;

	if (image.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
	{
		indexedImage.at<float>(y, x) = obj.Index;
		coloredImage.at<Vec3b>(y, x) = obj.Color;
		Recursion(image, indexedImage, coloredImage, y, x + 1,obj);
		Recursion(image, indexedImage, coloredImage, y + 1, x, obj);
		Recursion(image, indexedImage, coloredImage, y, x - 1, obj);
		Recursion(image, indexedImage, coloredImage, y - 1, x, obj);
	}
}

void ComputeCenterOfObjects(ObjectFeature &feature) {

		int index = 0;
		std::list<FeatureList>::iterator obj = feature.Objects.begin();
		std::list<FeatureList> reducedObjects;
		std::cout << "Computing centers" << std::endl;
		while (obj != feature.Objects.end())
		{
			float m10 = 0.0;
			float m01 = 0.0;
			float m00 = 0.0;
			for (int y = 0; y < feature.IndexedImage.rows; y++) {
				for (int x = 0; x < feature.IndexedImage.cols; x++) {
					if (feature.IndexedImage.at<float>(y, x) == (*obj).Index) {
						m10 += pow(x, 1) * pow(y, 0) * feature.IndexedImage.at<float>(y, x);
						m01 += pow(x, 0) * pow(y, 1) * feature.IndexedImage.at<float>(y, x);
						m00 += pow(x, 0) * pow(y, 0) * feature.IndexedImage.at<float>(y, x);
					}
				}
			}
			if (m00 > 100)
			{
				Point p = Point(m10 / m00, m01 / m00);
				(*obj).Center =  p;
				(*obj).Area = m00;
				reducedObjects.push_back(*obj);
			}

			obj++;
		}
		feature.Objects = reducedObjects;	
		std::cout << "Done ... " << std::endl;

}

double ComputePerimeterForObject(FeatureList &obj, Mat indexedImage, Mat coloredImage, int p, int q)
{
	double perimeter = 0.0;
	int index = obj.Index;
	Point center = obj.Center;
	for (int y = 0; y < indexedImage.rows; y++) {
		for (int x = 0; x < indexedImage.cols; x++) {


			if (indexedImage.at<float>(y, x) == index) {
				perimeter += pow((x - center.x), p) * pow((y - center.y), q) * index;
			}
		}
	}
	return perimeter;
}

void ComputePerimeter(ObjectFeature &feature)
{
	std::cout << "Computing perimeters" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		(*obj).Perimeter = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 0);
		obj++;
	}

	std::cout << "Done ..." << std::endl;

}

void ComputeFeatureOne(ObjectFeature &feature)
{
	std::cout << "Computing feature 1" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		(*obj).Feature1 = pow((*obj).Perimeter, 2) / (100 * (*obj).Area);
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}

void ComputeFeatureTwo(ObjectFeature &feature)
{
	std::cout << "Computing feature 2" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		double micro20 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 2, 0);
		double micro02 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 2);
		double micro11 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 1, 1);
		double microMax = (1 / 2) * (micro20 + micro02) + (1 / 2) * sqrt((4 * pow(micro11, 2)) + pow(micro20 - micro02, 2));
		double microMin = (1 / 2) * (micro20 + micro02) - (1 / 2) * sqrt((4 * pow(micro11, 2)) + pow(micro20 - micro02, 2));

		(*obj).Feature2 = microMin / microMax;
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}

void PutTextInimage(ObjectFeature &feature)
{
	std::cout << "Writing text to image" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		Point center = (*obj).Center;
		string feature1 = "F1: " + to_string((*obj).Feature1);
		string feature2 = "F2: " + to_string((*obj).Feature2);

		cv::putText(feature.ColoredImage,
			feature1,
			Point(center.x-10, center.y - 10), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			0.5, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255)); // BGR Color
		//std::cout << centerPoint<< std::endl;

		cv::putText(feature.ColoredImage,
			feature2,
			Point(center.x-10, center.y + 10), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			0.5, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255)); // BGR Color
		//std::cout << centerPoint<< std::endl;
		obj++;
	}
}

void ImageTresholding()
{
	Mat image = imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageGray;
	image.convertTo(imageGray, CV_32FC1, 1.0 / 255.0);
	Mat indexedImage = Mat::zeros(imageGray.size(), CV_32FC1);
	Mat coloredImage = Mat::zeros(imageGray.size(), CV_8UC3);

	std::list<FeatureList> objects;
	int index = rand() % 255 + 1;
	Vec3b color = Vec3b(rand() % 255, rand() % 255, rand() % 255);

	for (int y = 0; y < imageGray.rows; y++) {
		for (int x = 0; x < imageGray.cols; x++) {
			if (imageGray.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
			{
				FeatureList obj = FeatureList(index, color);
				Recursion(imageGray, indexedImage, coloredImage, y, x,obj);
				objects.push_back(obj);
				index = rand() % 255 + 1;
				color = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
			}
		}
	}

	ObjectFeature feature;
	feature.IndexedImage = indexedImage;
	feature.ColoredImage = coloredImage;
	feature.Objects = objects;

	ComputeCenterOfObjects(feature);
	ComputePerimeter(feature);
	ComputeFeatureOne(feature);
	ComputeFeatureTwo(feature);



	PutTextInimage(feature);
	imshow("Initial", imageGray);
	imshow("tresshold", feature.IndexedImage);
	imshow("color", feature.ColoredImage);

	cv::waitKey(0);

}


int main()
{
	ImageTresholding();
    
}

