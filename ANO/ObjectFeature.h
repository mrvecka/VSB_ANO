#pragma once
#include "pch.h"

using namespace cv;

class MyPoint {
public:
	double x;
	double y;
	MyPoint(double x, double y)
	{
		this->x = x;
		this->y = y;
	}
};

class Ethalon {
public:
	double x;
	double y;
	std::string label;
	Ethalon(double x, double y)
	{
		this->x = x;
		this->y = y;
	}

	Ethalon() {}

	void AddClass() {
		
		if (this->x >= 0.13 && this->x <= 0.16 && this->y >= 0.94 && this->y <= 0.98)
		{
			this->label = "Square";
		}
		else if (this->x >= 0.161  && this->x <= 0.21 && this->y >= 0.07 && this->y <= 0.11)
		{
			this->label = "Rectangle";
		}
		else if (this->x >= 0.62 && this->x <= 0.66 && this->y >= 0.89  && this->y <= 0.93)
		{
			this->label = "Star";
		}
		else
		{
			this->label = "Unknown";
		}

	}
};

class FeatureList {
public:
	double Index;
	double Area;
	Vec3b Color;
	Point Center;
	double Perimeter;
	double Feature1;
	double Feature2;
	Ethalon ClassLabel;

	FeatureList(int index, Vec3b color) {
		this->Index = index;
		this->Color = color;
	}

	static double GetPerimeter(FeatureList obj) { return obj.Perimeter; }
};

class ObjectFeature {
public:
	Mat IndexedImage;
	Mat ColoredImage;
	std::list<FeatureList> Objects;
	std::list<Ethalon> Ethalons;


};