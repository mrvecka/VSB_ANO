#pragma once
#include "pch.h"

using namespace cv;


class MyPoint {
public:
	double x;
	double y;
	MyPoint() {}
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
	int label;
	Ethalon(double x, double y,int lbl)
	{
		this->x = x;
		this->y = y;
		this->label = lbl;
	}

	Ethalon() {}
};

class FeatureList {
public:
	int Index;
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


class CentroidObject {
public:
	MyPoint Centroid;
	std::list<FeatureList> ClosestObjects;

	CentroidObject() {}
};

class ObjectFeature {
public:
	Mat IndexedImage;
	Mat ColoredImage;
	std::list<FeatureList> Objects;
	std::list<Ethalon> Ethalons;
	std::list<CentroidObject> Centroids;

};

