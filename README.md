Gesture-Recognition
===================

This is a desktop app to control VLC player functionality using hand gestures.

Assumptions:
- A webcam camera is connected, streaming live video feed.
- An instance of VLC player is already running when using gestures to control it.
- Before usage, hand gestures are trained and associated with a particular function such as Play/Pause/Next song.

Implementation details:
- The SURF keypoints and descriptors are extracted from each of the training images using OpenCV. These are used as features (number of dimensions = 64).
- Each of the extracted features is added to a "Bag of Words" model.
- A gesture dictionary is constructed after training all the supported gestures.
- SVM classifier is used to match a gesture from live input to one of the gestures in the dictionary.
