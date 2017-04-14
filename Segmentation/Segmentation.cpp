#include <cstdio>  
#include <vector>  
#include <iostream>  
#include <fstream>  
#include <cstring>
#include <map>
#include <algorithm> 
#include <sstream>
#include <stack>
#include "opencv\cv.h"
#include "opencv2\core\core.hpp"  
#include "opencv2\highgui\highgui.hpp"  
#include "opencv2\imgproc\imgproc.hpp"  
#include "opencv2\contrib\contrib.hpp" 
#include "Segmentation.h"
using namespace std;
using namespace cv;
int r = 1;
Segmentation::Segmentation(Mat srcimg) {
	srcImage = srcimg;
}
int Segmentation::hashVec3b(Vec3b &v) {
	int ret = 0;
	ret += v[0];
	ret <<= 8;
	ret += v[1];
	ret <<= 8;
	ret += v[2];
	return ret;
}

Vec3b Segmentation::rehashVec3b(int t) {
	int t1 = t >> 16;
	int t2 = (t >> 8) & 0x000000ff;
	int t3 = t & 0x000000ff;
	return Vec3b(t1, t2, t3);
}

Vec3b Segmentation::bgc(Mat srcimage) {
	map<int, int> m;
	int max = 0;
	int max1 = 0;
	int c = 0;
	for (int i = 0; i < srcimage.rows; i++) {
		for (int j = 0; j < srcimage.cols; j++) {
			int t = hashVec3b(srcimage.at<Vec3b>(i, j));
			m[t]++;
			if (max < m[t]) {
				max = m[t];
			}
			if (max1 < max && max1 < m[t]) {
				c = t;
			}
		}
	}
	return rehashVec3b(c);
}
Mat Segmentation::paint(Mat srcimg) {
	int width = srcimg.cols;
	int height = srcimg.rows;
	if (width > height) {
		Mat img(width, width, srcimg.type(), Scalar(0,0,0));
		int diff = width - height;
		int d = diff / 2;
		for (int i = d; i < width - d - (diff % 2); i++) {
			for (int j = 0; j < width; j++) {
				img.at<uchar>(i, j) = srcimg.at<uchar>(i - d, j);
			}
		}
		return img;
	}
	else if (width < height) {
		Mat img(height, height, srcimg.type(), Scalar(0, 0, 0));
		int diff = height - width;
		int d = diff / 2;
		for (int i = 0; i < height; i++) {
			for (int j = d; j < height - d - (diff % 2); j++) {
				img.at<uchar>(i, j) = srcimg.at<uchar>(i, j - d);
			}
		}
		return img;
	}
	return srcimg;

}
Mat Segmentation::paintVec3b(Mat srcimg) {
	int width = srcimg.cols;
	int height = srcimg.rows;
	if (width > height) {
		Mat img(width, width, srcimg.type(), Scalar(bgc(srcimg)));
		int diff = width - height;
		int d = diff / 2;
		for (int i = d; i < width-d-(diff%2); i++) {
			for (int j = 0; j < width; j++) {
				img.at<Vec3b>(i, j)[0] = srcimg.at<Vec3b>(i - d, j)[0];
				img.at<Vec3b>(i, j)[1] = srcimg.at<Vec3b>(i - d, j)[1];
				img.at<Vec3b>(i, j)[2] = srcimg.at<Vec3b>(i - d, j)[2];
			}
		}
		return img;
	}
	else if (width < height) {
		Mat img(height, height, srcimg.type(), Scalar(bgc(srcimg)));
		int diff = height - width;
		int d = diff / 2;
		for (int i = 0; i < height; i++) {
			for (int j = d; j < height - d - (diff % 2); j++) {
				img.at<Vec3b>(i, j)[0] = srcimg.at<Vec3b>(i, j - d)[0];
				img.at<Vec3b>(i, j)[1] = srcimg.at<Vec3b>(i, j - d)[1];
				img.at<Vec3b>(i, j)[2] = srcimg.at<Vec3b>(i, j - d)[2];
			}
		}
		return img;
	}
	return srcimg;

}
Mat  Segmentation::filter(Mat srcimage, int size, int num) {
	for (int i = size / 2; i < srcimage.rows - size / 2; i++) {
		for (int j = size / 2; j <srcimage.cols - size / 2; j++) {
			int count = 0;
			for (int f = -size / 2; f <= size / 2; f++) {
				for (int g = -size / 2; g <= size / 2; g++) {
					if (srcimage.at<uchar>(i + f, j + g) > 200) {
						count++;
					}
				}
			}
			if (count < num) {
				srcimage.at<uchar>(i, j) = 0;
			}
		}
	}
	return srcimage;
}
Mat Segmentation::getSrcImage() {
	return srcImage;
}
void Segmentation::setBinaryImage() {
	Mat grayimg;
	cvtColor(srcImage, grayimg, CV_BGR2GRAY);
	adaptiveThreshold(grayimg, binaryImage, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 33);
}
Mat Segmentation::getBinaryImage() {
	return binaryImage;
}
vector<Mat> Segmentation::getBinarySeg() {
	return binarySeg;
}
vector<Mat> Segmentation::getSrcSeg() {
	return srcSeg;
}
vector<Rect> Segmentation::getSegRect() {
	return  segRect;
}
Mat Segmentation::Dilation(Mat srcimg, int size)
{
	Mat dilatimg;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
	dilate(srcimg, dilatimg, element);
	return dilatimg;
}
Mat Segmentation::Erosion(Mat srcimg, int size) {
	Mat erodeimg;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
	erode(srcimg, erodeimg, element);
	return erodeimg;
}
void Segmentation::Slicimg(Mat srcimg, Mat yimg, Mat bimg, int cmin, int cmax) {
	Mat threshold_output = srcimg.clone();
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end()) {
		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		segRect.push_back(boundRect[i]);
	}
	/// 画多边形轮廓 + 包围的矩形框
	Mat drawing = bimg.clone();
	Mat src = bimg.clone();
	Mat ydrawing = yimg.clone();
	Mat ysrc = yimg.clone();
	vector<Mat> roi(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		/*stringstream stream, streams;
		string str,strs;
		stream << i;
		streams << r;
		stream >> str;
		streams >> strs;*/
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(ydrawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0);
		rectangle(ydrawing, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0);
		Mat temp(src, Rect(boundRect[i]));
		Mat ytemp(ysrc, Rect(boundRect[i]));
		Mat timg = paint(temp);
		Mat img;
		Mat yimg;
		resize(timg, img, Size(32, 32), 0, 0, CV_INTER_LINEAR);
		resize(ytemp, yimg, Size(32, 32), 0, 0, CV_INTER_LINEAR);
		binarySeg.push_back(img);
		srcSeg.push_back(yimg);
		//imshow("str", img);
	}
}

int main() {
	char buffer[40];
	for (r = 10; r < 20; r++) {
		sprintf(buffer, "C:\\Users\\yinmw\\Desktop\\picture\\%d.jpg", r);
		Mat srcimage = imread(buffer);
		Segmentation se(srcimage);
		stringstream stream;
		string str;
		stream << r;
		stream >> str;
		string filename = str + ".jpg";
		cout << filename << endl;
		se.setBinaryImage();
		Mat mimage = se.Erosion(se.getBinaryImage(), 2);
		Mat teimage = se.Dilation(mimage, 2);
		se.Slicimg(teimage, srcimage, teimage, 15, 300);
		vector<Mat> m = se.getBinarySeg();
		vector<Mat>::iterator it = m.begin();
		int k = 0;
		cout << m.size() << endl;
		for (it = m.begin(); it != m.end(); it++) {
			stringstream streams;
			string strs;
			streams << k;
			streams >> strs;
			imshow(strs, *it);
			++k;
		}
		cvWaitKey(0);
		/*Mat srcimage = imread(buffer);
		Mat grayimg;
		Mat seimg;
		Mat Mimage;
		cvtColor(srcimage, grayimg, CV_BGR2GRAY);
		adaptiveThreshold(grayimg, seimg, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 33);
		Mat mimage = Erosion(seimg, 2);
		Mat teimage = Dilation(mimage, 2);
		Mimage = Dilation(teimage, 1);
		stringstream stream;
		string str;
		stream << r;
		stream >> str;
		string filename = str + ".jpg";
		cout << filename << endl;
		Mat silimg = Slicimg(Mimage, srcimage, teimage, 15, 300);
		imshow("S", silimg);
		cvWaitKey(0);*/
	}
	//system("pause");
	return 0;
}

