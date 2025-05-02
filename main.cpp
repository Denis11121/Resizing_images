#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/proiect.h"
using namespace std;
using namespace cv;

int main() {

    Mat img = imread("C:/Users/Denis/Desktop/PI/proiect/images/cameraman.bmp",
                          IMREAD_GRAYSCALE);

    int newWidth = img.cols * 3;
    int newHeight = img.rows * 3;

    Mat resizedBilinear = resizeBilinear(img, newWidth, newHeight);

    imshow("Original", img);
    imshow("Redimensionat Biliniar", resizedBilinear);


    newWidth = img.cols * 3;
    newHeight = img.rows * 3;

    Mat resizedBicubic = resizeBicubic(img, newWidth, newHeight);
    imshow("Redimensionat Bicubic", resizedBicubic);

    newHeight=img.rows*3;
    newWidth=img.cols*3;
    Mat resizedNearestNeighbor = resizeNearestNeighbor(img, newWidth, newHeight);
    imshow("Redimensionat Nearest Neighbor", resizedNearestNeighbor);


    newWidth=img.cols*0.5;
    newHeight=img.rows*0.5;
    Mat resizedSubsamplingAverage = resizeSubsamplingAverage(img, newWidth, newHeight);
    imshow("Redimensionat Subsampling Average", resizedSubsamplingAverage);

    waitKey(0);
    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.