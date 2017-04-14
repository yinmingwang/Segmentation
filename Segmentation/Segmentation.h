#include <vector>  
#include <iostream>   
#include <cstring>
#include "opencv\cv.h"
#include "opencv2\core\core.hpp"  
#include "opencv2\highgui\highgui.hpp"  
#include "opencv2\imgproc\imgproc.hpp"  
#include "opencv2\contrib\contrib.hpp" 
using namespace std;
using namespace cv;

#ifndef SEGMENTATION
#define SEGMENTATION
class Segmentation{
public:
	Segmentation(Mat srcimg);
	int hashVec3b(Vec3b &v);
	Vec3b rehashVec3b(int t);
	Vec3b bgc(Mat srcimage);
	Mat paint(Mat srcimg);
	Mat  filter(Mat srcimage, int size, int num);
	Mat Dilation(Mat srcimg, int size);
	Mat Erosion(Mat srcimg, int size);
	void Slicimg(Mat srcimg, Mat yimg, Mat bimg, int cmin, int cmax);
	Mat paintVec3b(Mat srcimg);
	void setBinaryImage();
	Mat getBinaryImage();
	Mat getSrcImage();
	vector<Mat> getBinarySeg();
	vector<Mat> getSrcSeg();
	vector<Rect> getSegRect();
private:
	Mat srcImage;
	Mat binaryImage;
	vector<Mat> binarySeg;
	vector<Mat> srcSeg;
	vector<Rect> segRect;
};


#endif 