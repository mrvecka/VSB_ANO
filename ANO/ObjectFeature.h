#pragma once
#include "pch.h"

using namespace cv;

class FeatureList {
public:
	double Index;
	double Area;
	Vec3b Color;
	Point Center;
	double Perimeter;
	double Feature1;
	double Feature2;

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


};