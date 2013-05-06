/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#include "TrainGesture.h"

void TrainGesture::recordGesture(vector<Mat>& trainData, int frames_to_record)
{
    Mat frame;
    VideoCapture capture(1); // 0 : Built-in Webcam, 1 : Logitech Cam
    cout << "Enter 'r' to Start Recording Gesture or 'q' to Quit\n" << endl;
    int count_r = 0;
    bool record = false;

    if (!capture.isOpened())
    {
        cout << "Null Capture from Camera!\n";
    }

    for (;;)
    {
        capture >> frame;
        if (!frame.empty())
        {
            imshow("Camera Source", frame);

            if (cvWaitKey(10) == 'r') // will work only if a window is already open
            {
                cout << "Started Recording Gesture !" << endl;
                record = true;
            }

            if (record == true)
            {
                cvtColor(frame, frame, CV_BGR2GRAY);
                trainData.push_back(frame);
                count_r++;
                cout << "Recording frame : " << count_r << endl;
                if (count_r == frames_to_record)
                {
                    record = false;
                    cout << "Stopped Recording Gesture!\n" << endl;
                    destroyWindow("Camera Source");
                    break;
                }
            }
        }
        else
        {
            cout << " --(!) No captured frame -- Break!";
            break;
        }

        if (cvWaitKey(10) == 'q')
        {
            cout << "Quit Recording!" << endl;
            break;
        }
    }
}

void TrainGesture::playGesture(vector<Mat>& trainData)
{
    cout << "Enter 's' to start playing back Recorded Gesture!\n" << endl;
    namedWindow("Recorded Gesture", CV_WINDOW_KEEPRATIO);

    if (cvWaitKey(0) == 's')  // 0 because wait till eternity for s keypress
    {
        for (int i = 0; i < trainData.size(); i++)
        {
            imshow("Recorded Gesture", trainData[i]);
            if(waitKey(100) >= 0) break;
        }
    }
}
