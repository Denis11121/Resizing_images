//
// Created by Denis on 4/21/2025.
//

#ifndef PROIECT_H
#define PROIECT_H

#include <opencv2/opencv.hpp>


unsigned char bilinearInterpolate(cv::Mat img, float x, float y);
cv::Mat resizeBilinear(cv::Mat input, int newWidth, int newHeight);

unsigned char bicubicInterpolate(cv::Mat img, float x, float y);
cv::Mat resizeBicubic(cv::Mat input, int newWidth, int newHeight);

cv::Mat resizeNearestNeighbor(cv::Mat input, int newWidth, int newHeight);

cv::Mat resizeSubsamplingAverage(cv::Mat input, int newWidth, int newHeight);



#endif //PROIECT_H
