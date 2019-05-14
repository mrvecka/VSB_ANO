#include "pch.h"
#include "3DObjectDetection.h"

using namespace cv;
using namespace std;

void Test() {
	cv::Mat x_3x1 = cv::Mat::ones(3, 1, CV_64FC1);
	x_3x1.at<double>(0, 0) = -5.427;
	x_3x1.at<double>(1, 0) = 8.434;

	cv::Mat KR_3x3_inv = cv::Mat(3, 3, CV_64FC1);
	cv::Mat p_1x4 = cv::Mat(1, 4, CV_64FC1);
	cv::Mat C_3x1 = cv::Mat(3, 1, CV_64FC1);

	KR_3x3_inv = (cv::Mat_<double>(3, 3) << 0.048299, -0.00442434, -0.243132, 0.0129274, 0.0165135, 0.90747, 0.0000041464360679794796, 0.0469865, -0.341918);
	p_1x4 = (cv::Mat_<double>(1, 4) << 0, 0, 1, -20);
	C_3x1 = (cv::Mat_<double>(3, 1) << 10, -20, 15);

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

cv::Mat ImageToWorldSpace(cv::Mat y, cv::Mat P_3x3, cv::Mat P_3x1, cv::Mat normal, float d)
{

	cv::Mat P_3x3_inv = P_3x3.inv();

	cv::Mat C_3x1 = (-1 * P_3x3_inv) * P_3x1; //eye

	cv::Mat X_d_3x1 = P_3x3_inv * y;
	cv::Mat tmp1 = normal * C_3x1;
	double tmp1d = tmp1.at<double>(0, 0);

	cv::Mat tmp2 = normal * X_d_3x1;
	double tmp2d = tmp2.at<double>(0, 0);
	double lambda = -(tmp1d + d) / tmp2d;

	cout << "lambda " << endl << " " << lambda << endl;

	cv::Mat res = C_3x1 + lambda * X_d_3x1;
	cout << "d = " << endl << " " << d << endl;
	cout << "res = " << endl << " " << res << endl << endl;

	return res;
}

void Test2() {
	cv::Mat P_3x4 = cv::Mat::zeros(3, 4, CV_64FC1); // projection matrix

	cv::Mat rbl = cv::Mat::zeros(4, 1, CV_64FC1); // pojnt x in homogeneous coordinates
	cv::Mat rbr = cv::Mat::zeros(4, 1, CV_64FC1); // pojnt x in homogeneous coordinates
	cv::Mat fbl = cv::Mat::zeros(4, 1, CV_64FC1); // pojnt x in homogeneous coordinates
	cv::Mat fbr = cv::Mat::zeros(4, 1, CV_64FC1); // pojnt x in homogeneous coordinates

	//projection matrix from kitty calib file (projection matrix P2 because of left colored images)
	P_3x4 = (cv::Mat_<double>(3, 4) << 721.5377, 0, 609.5593, 44.85728, 0, 721.5377, 172.854, 0.2163791, 0, 0, 1, 0.002745884); 

	rbl = (cv::Mat_<double>(4, 1) << -1.895, 0, 0.775, 1); //rbl -> label 00013
	rbr = (cv::Mat_<double>(4, 1) << -1.895, 0, -0.775, 1); //rbl -> label 00013
	fbl = (cv::Mat_<double>(4, 1) << 1.895, 0, 0.775, 1); //fbl -> label 00013
	fbr = (cv::Mat_<double>(4, 1) << 1.895, 0, -0.775, 1); //fbr -> label 00013

	double rotation = 1.56f; // rotation around Y axis according to kitty label 00013
	cv::Mat RotationY_4x4 = (cv::Mat_<double>(4, 4) << cos(rotation), 0.0, sin(rotation), -3.59, 0.0, 1.0, 0.0, 1.69, -sin(rotation), 0.0, cos(rotation), 12.01, 0, 0, 0, 1); // rotation matrix

	P_3x4 = P_3x4 * RotationY_4x4;
	cv::Mat P_3x3 = P_3x4(cv::Rect(0,0,3,3)).clone(); //P3
	cv::Mat P_3x1 = P_3x4(cv::Rect(3, 0, 1, 3)).clone(); // fourth col of projection matrix

	// points in image plane
	cv::Mat rbl_3x1 = P_3x4 * rbl; 
	cv::Mat rbr_3x1 = P_3x4 * rbr;
	cv::Mat fbl_3x1 = P_3x4 * fbl;
	cv::Mat fbr_3x1 = P_3x4 * fbr;

	rbl_3x1 = rbl_3x1 / rbl_3x1.at<double>(2, 0);
	rbr_3x1 = rbr_3x1 / rbr_3x1.at<double>(2, 0);
	fbl_3x1 = fbl_3x1 / fbl_3x1.at<double>(2, 0);
	fbr_3x1 = fbr_3x1 / fbr_3x1.at<double>(2, 0);


	cout << "rbl_3x1 " << endl << " " << rbl_3x1 << endl << endl;
	cout << "rbr_3x1 " << endl << " " << rbr_3x1 << endl << endl;
	cout << "fbl_3x1 " << endl << " " << fbl_3x1 << endl << endl;
	cout << "fbr_3x1 " << endl << " " << fbr_3x1 << endl << endl;


	//projection from image plane to 3D world
	cv::Mat normal = (cv::Mat_<double>(1, 3) << 0, 1, 0);
	float d = 0.0f;
	cv::Mat fbl_res = ImageToWorldSpace(fbl_3x1, P_3x3, P_3x1, normal, d);
	cv::Mat rbl_res = ImageToWorldSpace(rbl_3x1, P_3x3, P_3x1, normal, d);
	cv::Mat fbr_res = ImageToWorldSpace(fbr_3x1, P_3x3, P_3x1, normal, d);
	cv::Mat rbr_res = ImageToWorldSpace(rbr_3x1, P_3x3, P_3x1, normal, d);


	cv::Mat tmp = rbl_res - fbl_res;
	//normal of front plane
	cv::Mat normal_front = tmp.reshape(0, 1);

	float d_f = -(normal_front.at<double>(0, 0)*fbl.at<double>(0, 0) + normal_front.at<double>(0, 1)*fbl.at<double>(1, 0) + normal_front.at<double>(0, 2)*fbl.at<double>(2, 0));

	//ftl in pixels
	cv::Mat ftl_3x1 = (cv::Mat_<double>(3, 1) << fbl_3x1.at<double>(0, 0), 185.42, 1);

	//ftl back to world space
	cv::Mat ftl_res = ImageToWorldSpace(ftl_3x1, P_3x3, P_3x1, normal_front, d_f);

	//points are 3x1 in world space
	cv::Mat top_vec = ftl_res - fbl_res;
	ftl_res = fbl_res + top_vec; 
	cv::Mat ftr_res = fbr_res + top_vec;
	cv::Mat rtl_res = rbl_res + top_vec;
	cv::Mat rtr_res = rbr_res + top_vec;

	//to homogeneous coordinates
	cv::Mat ftl_3d = (cv::Mat_<double>(4, 1) << ftl_res.at<double>(0, 0), ftl_res.at<double>(1, 0), ftl_res.at<double>(2, 0), 1);
	cv::Mat ftr_3d = (cv::Mat_<double>(4, 1) << ftr_res.at<double>(0, 0), ftr_res.at<double>(1, 0), ftr_res.at<double>(2, 0), 1);
	cv::Mat rtl_3d = (cv::Mat_<double>(4, 1) << rtl_res.at<double>(0, 0), rtl_res.at<double>(1, 0), rtl_res.at<double>(2, 0), 1);
	cv::Mat rtr_3d = (cv::Mat_<double>(4, 1) << rtr_res.at<double>(0, 0), rtr_res.at<double>(1, 0), rtr_res.at<double>(2, 0), 1);

	//point in image plane (pixels)
	cv::Mat ftl_image = P_3x4 * ftl_3d;
	cv::Mat ftr_image = P_3x4 * ftr_3d;
	cv::Mat rtl_image = P_3x4 * rtl_3d;
	cv::Mat rtr_image = P_3x4 * rtr_3d;

	ftl_image = ftl_image / ftl_image.at<double>(2, 0);
	ftr_image = ftr_image / ftr_image.at<double>(2, 0);
	rtl_image = rtl_image / rtl_image.at<double>(2, 0);
	rtr_image = rtr_image / rtr_image.at<double>(2, 0);

	cout << "ftl_image " << endl << " " << ftl_image << endl << endl;
	cout << "ftr_image " << endl << " " << ftr_image << endl << endl;
	cout << "rtl_image " << endl << " " << rtl_image << endl << endl;
	cout << "rtr_image " << endl << " " << rtr_image << endl << endl;

	Mat image = imread("C:\\Users\\Lukas\\Desktop\\semestralny projekt\\000871.png", CV_LOAD_IMAGE_COLOR);
	Point center_rbl_3x1 = Point(rbl_3x1.at<double>(0, 0), rbl_3x1.at<double>(1, 0));
	circle(image, center_rbl_3x1, 1, CV_RGB(255, 0, 0), 5);

	Point center_rbr_3x1 = Point(rbr_3x1.at<double>(0, 0), rbr_3x1.at<double>(1, 0));
	circle(image, center_rbr_3x1, 1, CV_RGB(255, 0, 0), 5);

	Point center_fbl_3x1 = Point(fbl_3x1.at<double>(0, 0), fbl_3x1.at<double>(1, 0));
	circle(image, center_fbl_3x1, 1, CV_RGB(255, 0, 0), 5);

	Point center_fbr_3x1 = Point(fbr_3x1.at<double>(0, 0), fbr_3x1.at<double>(1, 0));
	circle(image, center_fbr_3x1, 1, CV_RGB(255, 0, 0), 5);

	//front
	line(image, center_fbl_3x1, center_fbr_3x1, CV_RGB(0, 255, 0),2);

	//rear
	line(image, center_rbl_3x1, center_rbr_3x1, CV_RGB(255, 0, 0),2);

	//sides
	line(image, center_fbl_3x1, center_rbl_3x1, CV_RGB(0, 0, 255),2);
	line(image, center_fbr_3x1, center_rbr_3x1, CV_RGB(0, 0, 255),2);


	Point center_ftl_3x1 = Point(ftl_image.at<double>(0, 0), ftl_image.at<double>(1, 0));
	circle(image, center_ftl_3x1, 1, CV_RGB(255, 0, 0), 5);

	Point center_ftr_3x1 = Point(ftr_image.at<double>(0, 0), ftr_image.at<double>(1, 0));
	circle(image, center_ftr_3x1, 1, CV_RGB(255, 0, 0), 5);

	Point center_rtl_3x1 = Point(rtl_image.at<double>(0, 0), rtl_image.at<double>(1, 0));
	circle(image, center_rtl_3x1, 1, CV_RGB(255, 0, 0), 5);

	Point center_rtr_3x1 = Point(rtr_image.at<double>(0, 0), rtr_image.at<double>(1, 0));
	circle(image, center_rtr_3x1, 1, CV_RGB(255, 0, 0), 5);

	//front
	line(image, center_ftl_3x1, center_ftr_3x1, CV_RGB(0, 255, 0),2);

	//rear
	line(image, center_rtl_3x1, center_rtr_3x1, CV_RGB(255, 0, 0),2);

	//sides
	line(image, center_ftl_3x1, center_rtl_3x1, CV_RGB(0, 0, 255),2);
	line(image, center_ftr_3x1, center_rtr_3x1, CV_RGB(0, 0, 255), 2);


	//vertical
	line(image, center_ftl_3x1, center_fbl_3x1, CV_RGB(0, 255, 0), 2);
	line(image, center_ftr_3x1, center_fbr_3x1, CV_RGB(0, 255, 0), 2);

	line(image, center_rtl_3x1, center_rbl_3x1, CV_RGB(255, 0, 0), 2);
	line(image, center_rtr_3x1, center_rbr_3x1, CV_RGB(255, 0, 0), 2);

	//put vertex labels in image
	putText(image, "X_rbl", Point(center_rbl_3x1.x - 20, center_rbl_3x1.y + 20), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_rbr", Point(center_rbr_3x1.x - 20, center_rbr_3x1.y + 20), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_fbl", Point(center_fbl_3x1.x - 10, center_fbl_3x1.y + 20), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_fbr", Point(center_fbr_3x1.x - 10, center_fbr_3x1.y + 20), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_ftl", Point(center_ftl_3x1.x + 5, center_ftl_3x1.y + 15), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_ftr", Point(center_ftr_3x1.x - 10, center_ftr_3x1.y - 10), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_rtl", Point(center_rtl_3x1.x - 10, center_rtl_3x1.y - 10), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);
	putText(image, "X_rtr", Point(center_rtr_3x1.x - 10, center_rtr_3x1.y - 10), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2, LINE_AA);

	imshow("projected points", image);
	cv::waitKey(0);
}
