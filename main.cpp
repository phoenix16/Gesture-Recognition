/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#include "TrainGesture.h"
#include "GestureFeatures.h"

#define DICTIONARY_SIZE 100

int main()
{
    // Assumes already running instance of VLC player in dbus mode:
    // vlc --control dbus <playlist folder>

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

    GestureFeatures feat(DICTIONARY_SIZE, "SURF", "SURF");
    
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
            int response;
            feat.computeTestFeature(frame, response);

            switch(response)
            {
            case 1:
                cout << "Gesture 1 !" << endl;
                system("bash vlc-proxy.sh Play");
                break;
            case 2:
                cout << "Gesture 2 !" << endl;
                system("bash vlc-proxy.sh Pause");
                break;
            case 3:
                cout << "Gesture 3 !" << endl;
                system("bash vlc-proxy.sh Next");
                break;
            }

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


