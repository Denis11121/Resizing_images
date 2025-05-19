//
// Created by Denis on 4/21/2025.
//

#include "proiect.h"
#include <cmath>
using namespace cv;
using namespace std;

unsigned char bilinearInterpolate(Mat img, float x, float y) {

    int x1=floor(x);
    int y1=floor(y);

    int x2=std::min(x1+1, img.cols-1);
    int y2=std::min(y1+1, img.rows-1);

    float dx=x-x1;
    float dy=y-y1;

    float I1=(1-dx)*img.at<uchar>(y1,x1)+dx*img.at<uchar>(y1,x2);
    float I2=(1-dx)*img.at<uchar>(y2,x1)+dx*img.at<uchar>(y2,x2);

    return (uchar)((1-dy)*I1+dy*I2);
}

Mat resizeBilinear(Mat input, int newWidth, int newHeight) {
    Mat output(newHeight, newWidth, CV_8UC1);
    float scaleX=(float)input.cols/newWidth;
    float scaleY=(float)input.rows/newHeight;

    for (int y=0;y<newHeight;y++) {
        for (int x=0;x<newWidth;x++) {

            float srcX=x*scaleX;
            float srcY=y*scaleY;
            output.at<uchar>(y,x)=bilinearInterpolate(input,srcX,srcY);
        }
    }
    return output;
}

float cubicInterpolate(float p[4], float x) {
    //catmull-rom
    return p[1] + 0.5f * x * (p[2] - p[0] +
           x * (2*p[0] - 5*p[1] + 4*p[2] - p[3] +
           x * (3*(p[1] - p[2]) + p[3] - p[0])));
}

unsigned char bicubicInterpolate(Mat img, float x, float y) {
    int ix = floor(x);
    int iy = floor(y);

    float dx = x - ix;
    float dy = y - iy;

    float patch[4][4];

    for (int m = -1; m <= 2; m++) {
        int yy = iy + m;
        yy = max(0, min(yy, img.rows - 1));

        for (int n = -1; n <= 2; n++) {
            int xx = ix + n;
            xx = max(0, min(xx, img.cols - 1));

            patch[m + 1][n + 1] = img.at<uchar>(yy, xx);
        }
    }

    float col[4];
    for (int i = 0; i < 4; i++) {
        col[i] = cubicInterpolate(patch[i], dx);
    }

    float value = cubicInterpolate(col, dy);
    value = max(0.0f, min(255.0f, value));

    return (unsigned char)(value + 0.5f);
}


Mat resizeBicubic(Mat input, int newWidth, int newHeight) {
    Mat output(newHeight, newWidth, CV_8UC1);
    float scaleX = (float)input.cols / newWidth;
    float scaleY = (float)input.rows / newHeight;

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            output.at<uchar>(y, x) = bicubicInterpolate(input, srcX, srcY);
        }
    }
    return output;
}

Mat resizeNearestNeighbor(Mat input, int newWidth, int newHeight) {
    Mat output(newHeight, newWidth, CV_8UC1);

    float scaleX = (float)input.cols / newWidth;
    float scaleY = (float)input.rows / newHeight;

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            int srcX=(int)(x*scaleX+0.5f);
            int srcY=(int)(y*scaleY+0.5f);

            if (srcX>=input.cols) srcX=input.cols-1;
            if (srcY>=input.rows) srcY=input.rows-1;

            output.at<uchar>(y,x)=input.at<uchar>(srcY,srcX);
        }
    }
    return output;
}

Mat resizeSubsamplingAverage(Mat input, int newWidth, int newHeight) {
    Mat output(newHeight, newWidth, CV_8UC1);

    float scaleX = (float)input.cols / newWidth;
    float scaleY = (float)input.rows / newHeight;

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            int startX = (int)(x * scaleX);
            int startY = (int)(y * scaleY);
            int endX = (int)((x + 1) * scaleX);
            int endY = (int)((y + 1) * scaleY);

            endX = std::min(endX, input.cols);
            endY = std::min(endY, input.rows);

            int sum = 0;
            int count = 0;

            for (int j = startY; j < endY; j++) {
                for (int i = startX; i < endX; i++) {
                    sum += input.at<uchar>(j, i);
                    count++;
                }
            }

            uchar average = (uchar)(sum / count);
            output.at<uchar>(y, x) = average;
        }
    }


    return output;
}

//Mean Absolute Error
double calculateMAE(const Mat& img1, const Mat& img2) {
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    double sumError = 0.0;
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            sumError += abs(img1.at<uchar>(y, x) - img2.at<uchar>(y, x));
        }
    }
    return sumError / (img1.rows * img1.cols);
}

//Peak-Signal-to-Noise Ratio
double calculatePSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // convert to float
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum of all elements

    double mse = s.val[0] / (double)(I1.total()); //mean squared error
    if (mse == 0) return INFINITY;
    double psnr = 10.0 * log10((255 * 255) / mse);
    return psnr;
}