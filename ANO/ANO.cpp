// ANO.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "ObjectFeature.h"

#define M_PI 3.14159265358979323846
using namespace cv;
using namespace std;

void Recursion(Mat &image, Mat &indexedImage, Mat &coloredImage, int y, int x, FeatureList obj) 
{
	if (x > image.cols || x < 0)
		return;
	if (y > image.rows || y < 0)
		return;


	if (image.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
	{
		indexedImage.at<float>(y, x) = obj.Index;
		coloredImage.at<Vec3b>(y, x) = obj.Color;
		Recursion(image, indexedImage, coloredImage, y, x + 1, obj);
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

double ComputeEuclideanDistance(MyPoint a, MyPoint b)
{
	double distance = sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
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
	std::cout << "Getting class for objects" << std::endl;

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
	std::cout << "Done ..." << std::endl;

}

void ComputeKMeans(ObjectFeature &feature)
{
	std::cout << "Computing k-means" << std::endl;

	int numOfCentroids = feature.Ethalons.size();
	srand(time(NULL));
	std::list<CentroidObject> centroids;

	for (int i = 0; i < numOfCentroids; i++)
	{
		int index = rand() % (feature.Objects.size() - 1) + 1;
		std::list<FeatureList>::iterator it = feature.Objects.begin();
		std::advance(it, index);
		CentroidObject c;
		c.Centroid = MyPoint((*it).Feature1, (*it).Feature2);
		centroids.push_back(c);
	}
	double delta = 0.05;
	bool iterate = true;

	while (iterate)
	{
		//clear closestobject list from previous iteration
		std::list<CentroidObject>::iterator cen = centroids.begin();
		while (cen != centroids.end())
		{
			(*cen).ClosestObjects.clear();
			cen++;
		}

		//asign objects to centroids
		std::list<FeatureList>::iterator obj = feature.Objects.begin();
		while (obj != feature.Objects.end())
		{
			double distance = INFINITY;
			CentroidObject *closestCentroid = nullptr;
			MyPoint objectPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
			std::list<CentroidObject>::iterator cen = centroids.begin();
			while (cen != centroids.end())
			{
				double temp = ComputeEuclideanDistance((*cen).Centroid, objectPoint);
				if (temp < distance)
				{
					distance = temp;
					closestCentroid = &(*cen);
				}

				cen++;
			}
			closestCentroid->ClosestObjects.push_back((*obj));
			obj++;
		}

		int tmp = 0;
		//compute centroids
		cen = centroids.begin();
		while (cen != centroids.end())
		{
			if ((*cen).ClosestObjects.size() > 0)
			{
				MyPoint oldCentroid = (*cen).Centroid;
				double sumX = 0.0;
				double sumY = 0.0;
				int count = 0;
				std::list<FeatureList>::iterator obj = (*cen).ClosestObjects.begin();
				while (obj != (*cen).ClosestObjects.end())
				{
					sumX += (*obj).Feature1;
					sumY += (*obj).Feature2;
					count++;
					obj++;
				}

				MyPoint newCentroid = MyPoint(sumX / count, sumY / count);
				double dist = ComputeEuclideanDistance(oldCentroid, newCentroid);
				if (dist <= delta)
				{
					if (tmp == 0)
						iterate = false;
				}
				else
				{
					iterate = true;
				}

				(*cen).Centroid = newCentroid;

				tmp++;
			}
			
			cen++;
		}
	}
	
	feature.Centroids = centroids;
	std::cout << "Done ..." << std::endl;

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
		circle(coloredImage, cv::Point((*obj).Feature1*500, (*obj).Feature2*50), 3, cv::Scalar(0.0,0.0,255.0,1.0), 1);
		obj++;
	}

	std::list<CentroidObject>::iterator cen = feature.Centroids.begin();
	while (cen != feature.Centroids.end())
	{
		circle(coloredImage, cv::Point((*cen).Centroid.x * 500, (*cen).Centroid.y * 50), 6, cv::Scalar(255.0, 0.0, 0.0, 1.0), 1);
		cen++;
	}

	//imshow("Features ilustration", coloredImage);


}

ObjectFeature ImageTresholding()
{
	Mat image = imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat image = imread("images/test02.png", CV_LOAD_IMAGE_GRAYSCALE);

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
	ComputeKMeans(feature);

	IlustrateFeatures(feature);
	PutTextInimage(feature);

	/*imshow("Initial", imageGray);
	imshow("tresshold", feature.IndexedImage);
	imshow("color", feature.ColoredImage);*/

	//cv::waitKey(0);
	return feature;
}
void Test() {
	cv::Mat x_3x1 = cv::Mat::ones(3, 1, CV_64FC1);
	x_3x1.at<double>(0, 0) = -5.427;
	x_3x1.at<double>(1, 0) = 8.434;

	cv::Mat KR_3x3_inv = cv::Mat(3, 3, CV_64FC1);
	cv::Mat p_1x4 = cv::Mat(1, 4, CV_64FC1);
	cv::Mat C_3x1 = cv::Mat(3, 1, CV_64FC1);

	KR_3x3_inv = (cv::Mat_<double>(3,3) << 0.048299, -0.00442434, -0.243132, 0.0129274, 0.0165135, 0.90747, 0.0000041464360679794796, 0.0469865, -0.341918 );
	p_1x4 = (cv::Mat_<double>(1, 4) << 0,0,1,-20 );
	C_3x1 = (cv::Mat_<double>(3, 1) << 10,-20,15 );

	cout << "x_3x1 = " << endl << " " << x_3x1 << endl << endl;
	cout << "KR_3x3_inv = " << endl << " " << KR_3x3_inv << endl << endl;
	cout << "p_1x4 = " << endl << " " << p_1x4 << endl << endl;
	cout << "C_3x1 = " << endl << " " << C_3x1 << endl << endl;



	cv::Mat X_d_3x1 = KR_3x3_inv * x_3x1;
	cv::Mat tmp1 = p_1x4(cv::Rect(0, 0, 3, 1))*C_3x1;
	cv::Mat tmp2 = p_1x4(cv::Rect(0, 0, 3, 1))*X_d_3x1;
	double lambda = -(tmp1.at<double>(0, 0) + p_1x4.at<double>(0, 3)) / tmp2.at<double>(0, 0);

	cv::Mat res = C_3x1 + lambda * X_d_3x1;

	cout << "X_d_3x1 = " << endl << " " << X_d_3x1 << endl << endl;
	cout << "tmp1 = " << endl << " " << tmp1 << endl << endl;
	cout << "tmp2 = " << endl << " " << tmp2 << endl << endl;
	cout << "lambda = " << endl << " " << lambda << endl << endl;
	cout << "res = " << endl << " " << res << endl << endl;

}

void Test2() {
	cv::Mat P_3x4 = cv::Mat::zeros(3, 4, CV_64FC1); // projection matrix
	cv::Mat x_4x1 = cv::Mat::zeros(4, 1, CV_64FC1); // pojnt x in homogeneous coordinates

	P_3x4 = (cv::Mat_<double>(3, 4) << 721.5377, 0, 609.5593, 44.85728, 0, 721.5377, 172.854, 0.2163791, 0, 0, 1, 0.002745884);
	x_4x1 = (cv::Mat_<double>(4, 1) << 700, 223, 50, 1); //TODo:

	cv::Mat y_3x1 = cv::Mat::zeros(3, 1, CV_64FC1); // point x in image plane
	y_3x1 = P_3x4 * x_4x1;
	y_3x1 = y_3x1 / y_3x1.at<double>(2, 0);

	cout << "y_3x1 = " << endl << " " << y_3x1 << endl << endl;
	cv::Mat P_3x3 = cv::Mat::zeros(3, 3, CV_64FC1); //P3
	cv::Mat P_3x1 = cv::Mat::zeros(3, 1, CV_64FC1); // fourth col of projection matrix
	P_3x3 = (cv::Mat_<double>(3, 3) << 721.5377, 0, 609.5593, 44.85728, 0, 721.5377, 172.854, 0.2163791, 0);
	cv::Mat P_3x3_inv = P_3x3.inv();
	P_3x1 = (cv::Mat_<double>(3, 1) << 0, 1, 0.002745884);

	cv::Mat C_3x1 = cv::Mat::zeros(3, 1, CV_64FC1); //eye
	C_3x1 = -P_3x3.inv() * P_3x1;

	cv::Mat p_1x4 = cv::Mat(1, 4, CV_64FC1); 
	p_1x4 = (cv::Mat_<double>(1, 4) << 0, 1, 0, 50);

	cout << "y_3x1 = " << endl << " " << y_3x1 << endl << endl;
	cout << "P_3x3_inv = " << endl << " " << P_3x3_inv << endl << endl;
	cout << "p_1x4 = " << endl << " " << p_1x4 << endl << endl;
	cout << "C_3x1 = " << endl << " " << C_3x1 << endl << endl;

	cv::Mat X_d_3x1 = P_3x3_inv * y_3x1;
	cv::Mat tmp1 = p_1x4(cv::Rect(0, 0, 3, 1))*C_3x1;

	cv::Mat tmp2 = p_1x4(cv::Rect(0, 0, 3, 1))*X_d_3x1;
	double lambda = -(tmp1.at<double>(0, 0) + p_1x4.at<double>(0, 3)) / tmp2.at<double>(0, 0);

	cv::Mat res = C_3x1 + (P_3x3_inv * lambda) * y_3x1;

	cout << "X_d_3x1 = " << endl << " " << X_d_3x1 << endl << endl;
	cout << "tmp1 = " << endl << " " << tmp1 << endl << endl;
	cout << "tmp2 = " << endl << " " << tmp2 << endl << endl;
	cout << "lambda = " << endl << " " << lambda << endl << endl;
	cout << "res = " << endl << " " << res << endl << endl;
}

void train(NN* nn, ObjectFeature feature)
{
	//int n =1000;
	//double ** trainingSet = new double *[n];
	//for (int i = 0; i < n; i++) {
	//	trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

	//	bool classA = i % 2;

	//	for (int j = 0; j < nn->n[0]; j++) {
	//		if (classA) {
	//			trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
	//		}
	//		else {
	//			trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
	//		}
	//	}

	//	trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
	//	trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
	//}


	int n = feature.Objects.size();
	double ** trainingSet = new double *[n];

	int i = 0;
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
		for (int j = 0; j < nn->n[0]; j++) 
		{
			if(j%nn->n[0] == 0)
				trainingSet[i][j] = (*obj).Feature1;
			if(j%nn->n[0] == 1)
				trainingSet[i][j] = (*obj).Feature2;
			//add more features
		}

		if ((*obj).ClassLabel.label == "Square")
		{
			trainingSet[i][nn->n[0]] = 1.0;
			trainingSet[i][nn->n[0]+1] = 0.0;
			trainingSet[i][nn->n[0]+2] = 0.0;			
		}
		else if ((*obj).ClassLabel.label == "Rectangle")
		{
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 1.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}
		else if ((*obj).ClassLabel.label == "Star")
		{
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 1.0;
		}
		else
		{
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}

		i++;
		obj++;
	}


	double error = 1.0;
	i = 0;
	while (error > 0.1)
	{
		setInput(nn, trainingSet[i%n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i%n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
		//if (i == 100)
		//	break;
	}
	printf(" (%d iterations)\n", i);

	for (i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, ObjectFeature feature, int num_samples = 10)
{
	//double* in = new double[nn->n[0]];

	//int num_err = 0;
	//for (int n = 0; n < num_samples; n++)
	//{
	//	bool classA = rand() % 2;

	//	for (int j = 0; j < nn->n[0]; j++)
	//	{
	//		if (classA)
	//		{
	//			in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
	//		}
	//		else
	//		{
	//			in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
	//		}
	//	}
	//	printf("predicted: %d\n", !classA);
	//	setInput(nn, in, true);

	//	feedforward(nn);
	//	int output = getOutput(nn, true);
	//	if (output == classA) num_err++;
	//	printf("\n");
	//}
	
	int num_err = 0;

	double* in = new double[nn->n[0]];
	int i = 0;
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
 		for (int j = 0; j < nn->n[0]; j++)
		{
			if (j%nn->n[0] == 0)
				in[j] = (*obj).Feature1;
			if (j%nn->n[0] == 1)
				in[j] = (*obj).Feature2; 
			//add more features
		}
		int classA = 0;
		if ((*obj).ClassLabel.label == "Square")
			classA = 0;
		else if ((*obj).ClassLabel.label == "Rectangle")
			classA = 1;
		else if ((*obj).ClassLabel.label == "Star")
			classA = 2;
		else
			classA = 3;


		printf("predicted: %d\n", classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
		i++;
		obj++;
	}


	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}

int main(int argc, char** argv)
{
	ObjectFeature feature = ImageTresholding();
	//Test();
	Test2();



	/*NN * nn = createNN(2, 4, 3);
	train(nn,feature);

	getchar();

	test(nn,feature,100);

	getchar();

	releaseNN(nn);

	return 0;*/

}

