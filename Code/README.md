# simple_hand_gesture_recognition
Simple hand gesture recognition using opencv, sklearnkit and descriptions from the paper "A Simple and Effective Method for Hand Gesture Recognition""
Chetan Srinivasa Kumar 11839024
Shalini Maiti 11834150


To train your images, please run train.py with paths set appropriately in the file. The folder structure should be a root folder, with open_hand, v, three, and thumbs_up as subfolders with correct data. 
To test, run test.py with paths set appropriately with structure as specified above. 

WITHOUT setting paths, the code will simply utilize the training and test data in the Custom_test directory. FOR DEMO, directly run test.py to use the pretrained weights. 

PLEASE NOTE that once the knn weights are stored, it will be reused again in the next run. For new weights to be used, please delete older weights and then run test.py
