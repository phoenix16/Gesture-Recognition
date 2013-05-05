#ifndef TRAINGESTURE_H
#define TRAINGESTURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class TrainGesture
{
public:
    TrainGesture();
    void recordGesture(vector<Mat>& trainData, int frames_to_record);
    void playGesture(vector<Mat>& trainData);
};

#endif // TRAINGESTURE_H
