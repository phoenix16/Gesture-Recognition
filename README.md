Gesture-Recognition
===================

This is a desktop app to control VLC player functionality using hand gestures.

Assumptions:
- A webcam is connected to the computer, streaming live video feed.
- An instance of VLC player is running.
- Before using gestures to control the player functionality, hand gestures are trained and associated with a particular function such as Play/Pause/Next song.

Implementation:
- The SURF keypoints and descriptors are extracted from each of the training images using OpenCV. These are used as features (number of dimensions = 64).
- Each of the extracted features is added to a "Bag of Words" model.
- A gesture dictionary is constructed after training all the supported gestures.
- An SVM classifier is used to match a gesture from live video input to one of the gestures in the dictionary.
