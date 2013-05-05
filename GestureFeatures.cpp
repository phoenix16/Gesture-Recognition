#include "GestureFeatures.h"


// Bag of Words Implementation:

// Step 1: Extract the SURF local feature vectors from each of training images.
//         Put all the local feature vectors extracted into a single set,
//         doesn't matter which feature vector came from which training image
// Step 2: Apply a clustering algorithm (e.g. k-means) over the set of local feature vectors and find N centroid coordinates
//         for dictionary size N and assign an id to each centroid.
//         This set of centroids = BoW vocabulary
// Step 3: Find the nearest centroid for each local feature vector.
//         Global feature vector of each image = normalized histogram where
//         i-th bin of the histogram = frequency of i-th word of the vocabulary in the given image
//                                   = how many times ith centroid occurred in that image
// DictionarySize = number of centroids for K means clustering = number of bins in BoW histogram = size of global feature vector of image



// Constructor
GestureFeatures::GestureFeatures(int dictionarySize)
     :tc(CV_TERMCRIT_ITER, 10, 0.001),
      bowTrainer(dictionarySize, tc, 1, KMEANS_PP_CENTERS), // retries = 1
      SURFdetector(400),
      detector(FeatureDetector::create("FAST")), // SURF
      extractor(DescriptorExtractor::create("BRISK")), // SURF
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
// Find the global features and their labels for the training data ans train the SVM
void GestureFeatures::computeTrainFeatures(vector<Mat>& trainVector)
{
    cout << "Creating dictionary..." << endl;
    getTrainingVocabulary(trainVector);

    cout << "\nClustering " << bowTrainer.descripotorsCount() << " features to form dictionary..." << endl;
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
void GestureFeatures::computeTestFeature(Mat& testImage)
{
    vector<KeyPoint> keypoints;
    detector->detect(testImage, keypoints);
    bowDE.compute(testImage, keypoints, testData);
    cout << SVM.predict(testData) << endl;
}

