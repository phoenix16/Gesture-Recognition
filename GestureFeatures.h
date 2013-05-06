/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#ifndef GESTUREFEATURES_H
#define GESTUREFEATURES_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

class GestureFeatures
{
private:
    TermCriteria tc;
    BOWKMeansTrainer bowTrainer;
    SurfFeatureDetector SURFdetector;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Ptr<DescriptorMatcher> matcher;
    BOWImgDescriptorExtractor bowDE;
    CvSVMParams SVM_params;
    CvSVM SVM;    
    Mat trainData, trainLabels, testData;

    void getTrainingVocabulary(vector<Mat>& trainVector);
    void getBOWFeatures(vector<Mat>& inputVector, Mat& featureMat);
public:
    GestureFeatures(int dictionarySize);
    void computeTrainFeatures(vector<Mat>& trainVector);
    void computeTestFeature(Mat& testImage);
};

#endif // GESTUREFEATURES_H
