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

		std::list<FeatureList>::iterator obj = feature.Objects.begin();
		std::list<FeatureList> reducedObjects;
		std::cout << "Computing centers and perimeters" << std::endl;
		int perimeter = 0;
		while (obj != feature.Objects.end())
		{
			float m10 = 0.0;
			float m01 = 0.0;
			float m00 = 0.0;
			for (int y = 0; y < feature.IndexedImage.rows; y++) {
				for (int x = 0; x < feature.IndexedImage.cols; x++) {
					if (feature.IndexedImage.at<float>(y, x) == (*obj).Index) {
						m10 += pow(x, 1) * pow(y, 0);
						m01 += pow(x, 0) * pow(y, 1);
						m00 += pow(x, 0) * pow(y, 0);
						if (feature.IndexedImage.at<float>(y - 1, x) != (*obj).Index || feature.IndexedImage.at<float>(y + 1, x) != (*obj).Index || feature.IndexedImage.at<float>(y, x - 1) != (*obj).Index || feature.IndexedImage.at<float>(y, x + 1) != (*obj).Index)
						{
							perimeter++;
						}
					} 
				}
			}
			if (m00 > 100)
			{
				Point p = Point(m10 / m00, m01 / m00);
				(*obj).Center =  p;
				(*obj).Area = m00;
				(*obj).Perimeter = perimeter;
				reducedObjects.push_back(*obj);
			}
			perimeter = 0;
			obj++;
		}
		feature.Objects = reducedObjects;	
		std::cout << "Done ... " << std::endl;

}

int ComputeMicroForObject(FeatureList &obj, Mat indexedImage, Mat coloredImage, int p, int q)
{
	double micro = 0.0;
	int index = obj.Index;
	Point center = obj.Center;
	for (int y = 0; y < indexedImage.rows; y++) {
		for (int x = 0; x < indexedImage.cols; x++) {


			if (indexedImage.at<float>(y, x) == index) {
				micro += pow((x - center.x), p) * pow((y - center.y), q);

			}
		}
	}
	return micro;
}

//void ComputePerimeter(ObjectFeature &feature)
//{
//	std::cout << "Computing perimeters" << std::endl;
//
//	std::list<FeatureList>::iterator obj = feature.Objects.begin();
//	while (obj != feature.Objects.end())
//	{
//		(*obj).Feature1 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 0);
//		obj++;
//	}
//
//	std::cout << "Done ..." << std::endl;
//
//}

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
		double micro20 = ComputeMicroForObject((*obj), feature.IndexedImage, feature.ColoredImage, 2, 0);
		double micro02 = ComputeMicroForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 2);
		double micro11 = ComputeMicroForObject((*obj), feature.IndexedImage, feature.ColoredImage, 1, 1);
		double microMax = (1.0 / 2.0) * (micro20 + micro02) + (1.0 / 2.0) * sqrt((4 * pow(micro11, 2)) + pow(micro20 - micro02, 2));
		double microMin = (1.0 / 2.0) * (micro20 + micro02) - (1.0 / 2.0) * sqrt((4 * pow(micro11, 2)) + pow(micro20 - micro02, 2));

		(*obj).Feature2 = microMin / microMax;
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}

double ComputeEuclideanDistance(MyPoint a, Ethalon b)
{
	double distance = sqrt( pow(b.x-a.x,2) + pow(b.y - a.y,2) );
	return distance;
}

void ComputeEthalons(ObjectFeature &feature)
{
	std::cout << "Computing  ethalons" << std::endl;

	double x = 0.0;
	double y = 0.0;
	std::list<Ethalon> ethalons;
	Ethalon currentEthalon = Ethalon(0.0,0.0);
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		//feature1 => x
		//feature2 => y

		if (currentEthalon.x == 0.0)
		{
			currentEthalon = Ethalon((*obj).Feature1, (*obj).Feature2);
			ethalons.push_back(currentEthalon);
		}
		else
		{
			MyPoint currentPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
			std::list<Ethalon>::iterator eth = ethalons.begin();
			bool found = false;
			while (eth != ethalons.end())
			{
				if (ComputeEuclideanDistance(currentPoint, (*eth)) < 0.2)
				{
					(*eth) = Ethalon((currentPoint.x + (*eth).x) / 2, (currentPoint.y + (*eth).y) / 2);
					found = true;
				}

				eth++;
			}
			if (!found)
			{
				ethalons.push_back(Ethalon(currentPoint.x,currentPoint.y));
			}
		}
		obj++;
	}
	std::list<Ethalon>::iterator eth = ethalons.begin();
	while (eth != ethalons.end())
	{
		(*eth).AddClass();
		eth++;
	}

	feature.Ethalons = ethalons;
	std::cout << "Done ..." << std::endl;

}

void AssignClassToObject(ObjectFeature &feature)
{
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		MyPoint objectPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
		double mindst = INFINITY;
		Ethalon closestEthalon;
		std::list<Ethalon>::iterator eth = feature.Ethalons.begin();
		while (eth != feature.Ethalons.end())
		{
			double dst = ComputeEuclideanDistance(objectPoint, (*eth));
			if (dst < mindst)
			{
				mindst = dst;
				closestEthalon = (*eth);
			}
			eth++;
		}
		(*obj).ClassLabel = closestEthalon;
		obj++;
	}
}

void PutTextInimage(ObjectFeature &feature)
{
	std::cout << "Writing text to image" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		Point center = (*obj).Center;
		stringstream stream;
		stream << fixed << setprecision(2) << (*obj).Perimeter;
		string feature1 = "F1:" + stream.str();
		stream.str("");
		stream << fixed << setprecision(2) << (*obj).Area;
		string feature2 = "F2:" + stream.str();

		string text = (*obj).ClassLabel.label;

		cv::putText(feature.ColoredImage,
			text,
			Point(center.x, center.y), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			0.5, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255)); // BGR Color
		//std::cout << centerPoint<< std::endl;

		//cv::putText(feature.ColoredImage,
		//	feature2,
		//	Point(center.x-10, center.y + 5), // Coordinates
		//	cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
		//	0.5, // Scale. 2.0 = 2x bigger
		//	cv::Scalar(255, 255, 255)); // BGR Color
		////std::cout << centerPoint<< std::endl;
		obj++;
	}
}

void IlustrateFeatures(ObjectFeature &feature)
{
	Mat coloredImage = Mat::zeros(cv::Size(600,200), CV_8UC3);
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		circle(coloredImage, cv::Point((*obj).Feature1*500, (*obj).Feature2*50), 3, (255, 255, 255), 1);
		obj++;
	}
	imshow("Features ilustration", coloredImage);


}

void ImageTresholding()
{
	Mat image = imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageGray;
	image.convertTo(imageGray, CV_32FC1, 1.0 / 255.0);
	Mat indexedImage = Mat::zeros(imageGray.size(), CV_32FC1);
	Mat coloredImage = Mat::zeros(imageGray.size(), CV_8UC3);

	std::list<FeatureList> objects;
	int index = 1;
	Vec3b color = Vec3b(rand() % 255, rand() % 255, rand() % 255);

	for (int y = 0; y < imageGray.rows; y++) {
		for (int x = 0; x < imageGray.cols; x++) {
			if (imageGray.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
			{
				FeatureList obj = FeatureList(index, color);
				Recursion(imageGray, indexedImage, coloredImage, y, x,obj);
				objects.push_back(obj);
				index += 1;
				color = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
			}
		}
	}

	ObjectFeature feature;
	feature.IndexedImage = indexedImage;
	feature.ColoredImage = coloredImage;
	feature.Objects = objects;

	ComputeCenterOfObjects(feature);
	//ComputePerimeter(feature);
	ComputeFeatureOne(feature);
	ComputeFeatureTwo(feature);
	ComputeEthalons(feature);
	AssignClassToObject(feature);

	IlustrateFeatures(feature);
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

