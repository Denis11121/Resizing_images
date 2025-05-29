//
// Created by Denis on 4/21/2025.
//

#ifndef PROIECT_H
#define PROIECT_H

#include <opencv2/opencv.hpp>

//--grayscale--
unsigned char bilinearInterpolate(cv::Mat img, float x, float y);
cv::Mat resizeBilinear(cv::Mat input, int newWidth, int newHeight);

unsigned char bicubicInterpolate(cv::Mat img, float x, float y);
cv::Mat resizeBicubic(cv::Mat input, int newWidth, int newHeight);

cv::Mat resizeNearestNeighbor(cv::Mat input, int newWidth, int newHeight);

cv::Mat resizeSubsamplingAverage(cv::Mat input, int newWidth, int newHeight);

double calculateMAE(const cv::Mat& img1, const cv::Mat& img2);

double calculatePSNR(const cv::Mat& I1, const cv::Mat& I2);

//--color--
cv::Mat resizeBilinearColor(const cv::Mat& input, int newWidth, int newHeight);
cv::Mat resizeBicubicColor(const cv::Mat& input, int newWidth, int newHeight);
cv::Mat resizeNearestNeighborColor(const cv::Mat& input, int newWidth, int newHeight);
cv::Mat resizeSubsamplingAverageColor(const cv::Mat& input, int newWidth, int newHeight);

double calculateMAEColor(const cv::Mat& img1, const cv::Mat& img2);
double calculatePSNRColor(const cv::Mat& img1, const cv::Mat& img2);

#endif //PROIECT_H
