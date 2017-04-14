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
using namespace std;
using namespace cv;
int r = 1;
//FILE *fp = fopen("C:\\Users\\yinmw\\Desktop\\zt.txt", "w");
int hashVec3b(Vec3b &v) {
	int ret = 0;
	ret += v[0];
	ret <<= 8;
	ret += v[1];
	ret <<= 8;
	ret += v[2];
	return ret;
}

Vec3b rehashVec3b(int t) {
	int t1 = t >> 16;
	int t2 = (t >> 8) & 0x000000ff;
	int t3 = t & 0x000000ff;
	return Vec3b(t1, t2, t3);
}

Vec3b bgc(Mat srcimage) {
	map<int, int> m;
	int max = 0;
	int max1 = 0;
	int c = 0;
	for (int i = 0; i < srcimage.rows; i++) {
		for (int j = 0; j < srcimage.cols; j++) {
			//cout << srcimage.at<Vec3b>(i, j) << endl;
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
	//cout << max << endl;
	//cout <<rehashVec3b(c) << endl;
	return rehashVec3b(c);
}
Mat paint(Mat srcimg) {
	int width = srcimg.cols;
	int height = srcimg.rows;
	if (width > height) {
		Mat img(width, width, srcimg.type(), Scalar(0,0,0));
		//imshow("img", img);
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
		//imshow("img", img);
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
/*Mat paint(Mat srcimg) {
	//Mat img = srcimg.clone();
	int width = srcimg.cols;
	int height = srcimg.rows;
	if (width > height) {
		Mat img(width, width, srcimg.type(), Scalar(bgc(srcimg)));
		//imshow("img", img);
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
		//imshow("img", img);
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

}*/
Mat  filter(Mat srcimage, int size, int num) {
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
Mat Dilation(Mat srcimg, int size)
{
	Mat dilatimg;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
	dilate(srcimg, dilatimg, element);
	return dilatimg;
}
Mat Erosion(Mat srcimg, int size) {
	Mat erodeimg;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
	/// ¸¯Ê´²Ù×÷
	erode(srcimg, erodeimg, element);
	//imshow("Erosion Demo", erodeimg);
	return erodeimg;
}



Mat Slicimg(Mat srcimg, Mat yimg, Mat bimg, int cmin, int cmax) {
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
	}


	/// »­¶à±ßÐÎÂÖÀª + °üÎ§µÄ¾ØÐÎ¿ò
	//Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing = bimg.clone();
	Mat src = bimg.clone();
	Mat ydrawing = yimg.clone();
	Mat ysrc = yimg.clone();
	vector<Mat> roi(contours.size());
	/*if (contours.size() < 4 || contours.size() > 4) {
		cout << contours.size() << endl;
		fprintf_s(fp, "%d\n", r);
	}*/
	for (int i = 0; i < contours.size(); i++)
	{
		stringstream stream, streams;
		string str,strs;
		stream << i;
		streams << r;
		stream >> str;
		streams >> strs;
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
		//imshow(str, img);
		string path = "C:\\Users\\yinmw\\Desktop\\zt_binarybg\\" + strs + "_" + str + ".jpg";
		string path2 = "C:\\Users\\yinmw\\Desktop\\zt_srcbg\\" + strs + "_" + str + ".jpg";
		imwrite(path, img);
		imwrite(path2, yimg);
	}
	return drawing;
}

int main() {
	char buffer[40];
	for (r = 1; r < 20000; r++) {
		sprintf(buffer, "C:\\Users\\yinmw\\Desktop\\pic\\%d.jpg", r);
		Mat srcimage = imread(buffer);
		Mat grayimg;
		Mat seimg;
		Mat Mimage;
		cvtColor(srcimage, grayimg, CV_BGR2GRAY);
		//seimg = otsu(grayimg);
		adaptiveThreshold(grayimg, seimg, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 33);
		Mimage = Dilation(seimg, 3);
		///Mat s = filter(seimg, 7, 5);
		//medianBlur(seimg, Mimage, 3);
		//Mimage = Erosion(seimg, 2);
		//Mimage = Dilation(Mimage, 2);
		//imshow("f", Mimage);
		//
		stringstream stream;
		string str;
		stream << r;
		stream >> str;
		string filename = str + ".jpg";
		cout << filename << endl;
		string result = "C:\\Users\\yinmw\\Desktop\\zt\\" + filename;
		//imshow("src", srcimage);
		imwrite(result, seimg);
		Mat silimg = Slicimg(Mimage, srcimage, seimg, 15, 300);
		//imshow("S", silimg);
		//vWaitKey(0);
	}
	//fclose(fp);
	return 0;
}

