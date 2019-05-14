#pragma once
#include "pch.h"

void Test();
void Test2();

cv::Mat ImageToWorldSpace(cv::Mat y, cv::Mat P_3x3, cv::Mat P_3x1, cv::Mat normal, float d);