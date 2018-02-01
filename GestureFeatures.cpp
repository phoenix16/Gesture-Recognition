//
// Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
//

#include "GestureFeatures.h"

// Bag of Words Implementation:

// Constructor
GestureFeatures::GestureFeatures(int dictionarySize, string detector_type, string descriptor_type)
     :tc(CV_TERMCRIT_ITER, 10, 0.001),
      bowTrainer(dictionarySize, tc, 1, KMEANS_PP_CENTERS), // retries = 1
      SURFdetector(400),
      detector(FeatureDetector::create(detector_type)),
      extractor(DescriptorExtractor::create(descriptor_type)),
      matcher(DescriptorMatcher::create("FlannBased")),
      bowDE(extractor, matcher),
      trainData(0, dictionarySize, CV_32FC1),
      trainLabels(0, 1, CV_32FC1),
      testData(0, dictionarySize, CV_32FC1)
{
}

// Private function
// Extracts features from the training images and adds them to the bowTrainer
void GestureFeatures::getTrainingVocabulary(vector<Mat>& trainVector)
{
    int count = 0;
    for (vector<Mat>::iterator it = trainVector.begin(); it != trainVector.end(); it++)
    {
        Mat image = *it;
        if (!image.empty())
        {
            count++;
            cout << "\nProcessing training gesture " << count << "...";
            // Detect the Keypoints
            vector<KeyPoint> keypoints;
            detector->detect(image, keypoints);
            // SURFdetector.detect(image, keypoints);
            cout << "\tFound " << keypoints.size() << " keypoints" << endl;

            if (keypoints.empty())
            {
                cerr << "Warning: Could not find keypoints in image: " << count << endl;
            }
            else
            {
                // Get the Descriptors
                Mat features;
                extractor->compute(image, keypoints, features);   // features dim = 64 for SURF
                bowTrainer.add(features);           // throw each feature vector into the bag
            }
        }
        else
        {
            cerr << "Warning: Could not read image: " << count << endl;
        }
    }
}


// Private function
// Creates a BoW global feature vector (normalized histogram) for each image encountered.
// After the dictionary has been constructed, images (training or test) can be described by extracting
// features from them and matching them with the features in the dictionary which are closest.
void GestureFeatures::getBOWFeatures(vector<Mat>& inputVector, Mat& featureMat)
{
    int count = 0;
    for (vector<Mat>::iterator it = inputVector.begin(); it != inputVector.end(); it++)
    {
        Mat image = *it;
        if (!image.empty())
        {
            count++;
            cout << "\nProcessing training gesture " << count << "...";
            // Detect the SURF Keypoints
            vector<KeyPoint> keypoints;
            detector->detect(image, keypoints);
            // SURFdetector.detect(image, keypoints);
            cout << "\tFound " << keypoints.size() << " keypoints" << endl;

            if (keypoints.empty())
            {
                cerr << "Warning: Could not find keypoints in image: " << count << endl;
            }
            else
            {
                Mat bowFeature; float label = float(count);
                // does NOT compute SURF descriptors, finds global feature vector of image by finding nearest centroid, normalized histogram etc..
                bowDE.compute(image, keypoints, bowFeature);
                featureMat.push_back(bowFeature);

                trainLabels.push_back(label);
            }
        }
        else
        {
            cerr << "Warning: Could not read image: " << count << endl;
        }
    }
}


// Public function
// Find the global features and their labels for the training data and train the SVM
void GestureFeatures::computeTrainFeatures(vector<Mat>& trainVector)
{
    cout << "Creating dictionary..." << endl;
    getTrainingVocabulary(trainVector);

    cout << "\nClustering " << bowTrainer.descriptorsCount() << " features to form dictionary..." << endl;
    Mat dictionary = bowTrainer.cluster();
    bowDE.setVocabulary(dictionary);
    // Dictionary contains the centroids of the training set features. This is NOT the training data to be fed into the classifier
    cout << "\nDictionary size = [Number of Centroids]x[Feature Dimension] = " << dictionary.rows << " x " << dictionary.cols << endl;

    cout << "\nProcessing Training data..." << endl;
    getBOWFeatures(trainVector, trainData);
    cout << "\nTraining Data size = [Number of Training images]x[Dictionary size] = " << trainData.rows << " x " << trainData.cols << endl;

    cout << "\nTraining SVM Classifier..." << endl;
    SVM.train(trainData, trainLabels, Mat(), Mat(), SVM_params);
}


// Public function
// Find the global feature vector for the test data and predict it using SVM
void GestureFeatures::computeTestFeature(Mat& testImage, int& response)
{
    vector<KeyPoint> keypoints;
    detector->detect(testImage, keypoints);
    bowDE.compute(testImage, keypoints, testData);
    response = (int)SVM.predict(testData);
}

