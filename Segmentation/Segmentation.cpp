#include <cstdio>  
#include <vector>  
#include <iostream>  
#include <fstream>  
#include <cstring>   
#include <algorithm> 
#include <sstream>
#include "opencv\cv.h"  
#include "opencv2\core\core.hpp"  
#include "opencv2\highgui\highgui.hpp"  
#include "opencv2\imgproc\imgproc.hpp"  
#include "opencv2\contrib\contrib.hpp" 
using namespace std;
using namespace cv;
#define size 5
int main() {
	char buffer[40];
	for (int k = 1; k <= 200; k++) {
		sprintf(buffer, "C:\\Users\\yinmw\\Desktop\\picture\\%d.jpg", k);
		Mat srcimage = imread(buffer);
		Mat grayimg;
		Mat seimg;
		Mat Mimage;
		cvtColor(srcimage, grayimg, CV_BGR2GRAY);
		adaptiveThreshold(grayimg, seimg, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 17,33);
		//bilateralFilter(seimg, Mimage, 9, 9 * 2, 9 / 2);
		//medianBlur(seimg, Mimage, 3);
		for (int i = size / 2; i < seimg.rows - size / 2; i++) {
			for (int j = size / 2; j < seimg.cols - size / 2; j++) {
				int count = 0;
				for (int f = -size / 2; f <= size/2; f++) {
					for (int g = -size / 2; g <= size / 2; g++) {
						if (seimg.at<uchar>(i + f, j + g) > 200) {
							count++;
						}
					}
				}
				if (count < 5) {
					seimg.at<uchar>(i, j) = 0;
				}
			}
		}
		medianBlur(seimg, Mimage, 3);
		stringstream stream;
		string str;
		stream << k;
		stream >> str;
		string filename = str + ".jpg";
		cout << filename << endl;
		string result = "C:\\Users\\yinmw\\Desktop\\re\\" + filename;
		//imshow("src", srcimage);
		//imshow(result, Mimage);
		imwrite(result, Mimage);
		//cvWaitKey(0);
	}
	return 0;
}