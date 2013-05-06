/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#include "TrainGesture.h"
#include "GestureFeatures.h"

#define DICTIONARY_SIZE 10

int main()
{
    TrainGesture g_obj;

    int numGestures = 3;
    vector<Mat> trainVector;    // vector<Mat> of training Set of gestures. Each element is a unique gesture
    for (int i = 0; i < numGestures; i++)
    {
        vector<Mat> trainFrames;
        g_obj.recordGesture(trainFrames, 10);  // Collect 10 frames of each gesture
        trainVector.push_back(trainFrames[5].clone());   // Store only one frame per gesture as training image
//        g_obj.playGesture(trainFrames);
    }

    GestureFeatures feat(DICTIONARY_SIZE);
    feat.computeTrainFeatures(trainVector);

    Mat frame;
    VideoCapture capture(1); // 0 : Built-in Webcam, 1 : Logitech Cam

    if (!capture.isOpened())
    {
        cout << "Null Capture from Camera!\n";
        return 0;
    }

    for (;;)
    {
        capture >> frame;
        if (!frame.empty())
        {
            imshow("input", frame);
            cvtColor(frame, frame, CV_BGR2GRAY);

            // Find the global feature vector of test frame and classify the gesture
            feat.computeTestFeature(frame);
        }
        else
        {
            cout << " --(!) No captured frame -- Break!";
            break;
        }

        if (cvWaitKey(10) == 27)  // ESC to exit
            break;
    }

    return(0);
}


