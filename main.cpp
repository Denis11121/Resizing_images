#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/proiect.h"
using namespace std;
using namespace cv;

int main() {

    Mat img = imread("C:/Users/Denis/Desktop/PI/proiect/images/cameraman.bmp",
                     IMREAD_GRAYSCALE);

    if (img.empty()) {
        cerr << "Imaginea nu a putut fi incarcata!" << endl;
        return -1;
    }

    int newWidth = img.cols * 3;
    int newHeight = img.rows * 2;

    //Bilinear resize + timing
    int64 t1 = getTickCount();
    Mat resizedBilinear = resizeBilinear(img, newWidth, newHeight);
    int64 t2 = getTickCount();
    double timeBilinear = (t2 - t1) * 1000.0 / getTickFrequency();  // ms


    imshow("Original", img);
    imshow("Redimensionat Biliniar", resizedBilinear);

    //Bicubic resize + timing
    newWidth = img.cols * 3;
    newHeight = img.rows * 3;
    t1 = getTickCount();
    Mat resizedBicubic = resizeBicubic(img, newWidth, newHeight);
    t2 = getTickCount();
    double timeBicubic = (t2 - t1) * 1000.0 / getTickFrequency();


    imshow("Redimensionat Bicubic", resizedBicubic);

    // Nearest Neighbor resize + timing
    newHeight = img.rows * 3;
    newWidth = img.cols * 3;
    t1 = getTickCount();
    Mat resizedNearestNeighbor = resizeNearestNeighbor(img, newWidth, newHeight);
    t2 = getTickCount();
    double timeNearest = (t2 - t1) * 1000.0 / getTickFrequency();


    imshow("Redimensionat Nearest Neighbor", resizedNearestNeighbor);

    //Subsampling Average resize + timing
    newWidth = (int)(img.cols / 1.4);
    newHeight = (int)(img.rows / 1.4);
    t1 = getTickCount();
    Mat resizedSubsamplingAverage = resizeSubsamplingAverage(img, newWidth, newHeight);
    t2 = getTickCount();
    double timeSubsampling = (t2 - t1) * 1000.0 / getTickFrequency();


    imshow("Redimensionat Subsampling Average", resizedSubsamplingAverage);

    cout << "Timp executie Bilinear: " << timeBilinear << " ms" << endl;
    cout << "Timp executie Bicubic: " << timeBicubic << " ms" << endl;
    cout << "Timp executie Nearest Neighbor: " << timeNearest << " ms" << endl;
    cout << "Timp executie Subsampling Average: " << timeSubsampling << " ms" << endl;


    // OpenCV resize results
    Mat cvBilinear, cvBicubic, cvNearest, cvSubsampling;

    // Resize si timp cu metode OpenCV pentru comparatie
     // OpenCV resize results + timing
     int64 t1_ocv, t2_ocv;
     double timeOCV;

     // OpenCV Bilinear
     t1_ocv = getTickCount();
     resize(img, cvBilinear, Size(resizedBilinear.cols, resizedBilinear.rows), 0, 0, INTER_LINEAR);
     t2_ocv = getTickCount();
     timeOCV = (t2_ocv - t1_ocv) * 1000.0 / getTickFrequency();
     cout << "\n---Timp OpenCV---\nTimp executie OpenCV Bilinear: " << timeOCV << " ms" << endl;

     // OpenCV Bicubic
     t1_ocv = getTickCount();
     resize(img, cvBicubic, Size(resizedBicubic.cols, resizedBicubic.rows), 0, 0, INTER_CUBIC);
     t2_ocv = getTickCount();
     timeOCV = (t2_ocv - t1_ocv) * 1000.0 / getTickFrequency();
     cout << "Timp executie OpenCV Bicubic: " << timeOCV << " ms" << endl;

     // OpenCV Nearest Neighbor
     t1_ocv = getTickCount();
     resize(img, cvNearest, Size(resizedNearestNeighbor.cols, resizedNearestNeighbor.rows), 0, 0, INTER_NEAREST);
     t2_ocv = getTickCount();
     timeOCV = (t2_ocv - t1_ocv) * 1000.0 / getTickFrequency();
     cout << "Timp executie OpenCV Nearest Neighbor: " << timeOCV << " ms" << endl;

     // OpenCV Subsampling (INTER_AREA)
     t1_ocv = getTickCount();
     resize(img, cvSubsampling, Size(resizedSubsamplingAverage.cols, resizedSubsamplingAverage.rows), 0, 0, INTER_AREA);
     t2_ocv = getTickCount();
     timeOCV = (t2_ocv - t1_ocv) * 1000.0 / getTickFrequency();
     cout << "Timp executie OpenCV Subsampling: " << timeOCV << " ms" << endl;

    // MSE si PSNR
    cout << "\n--- Comparatie cu OpenCV ---" << endl;
    cout << "Bilinear - MSE: " << calculateMAE(resizedBilinear, cvBilinear)
         << ", PSNR: " << calculatePSNR(resizedBilinear, cvBilinear) << " dB" << endl;

    cout << "Bicubic - MSE: " << calculateMAE(resizedBicubic, cvBicubic)
         << ", PSNR: " << calculatePSNR(resizedBicubic, cvBicubic) << " dB" << endl;

    cout << "Nearest Neighbor - MSE: " << calculateMAE(resizedNearestNeighbor, cvNearest)
         << ", PSNR: " << calculatePSNR(resizedNearestNeighbor, cvNearest) << " dB" << endl;

    cout << "Subsampling Average - MSE: " << calculateMAE(resizedSubsamplingAverage, cvSubsampling)
         << ", PSNR: " << calculatePSNR(resizedSubsamplingAverage, cvSubsampling) << " dB" << endl;

    waitKey(0);
    return 0;
}
