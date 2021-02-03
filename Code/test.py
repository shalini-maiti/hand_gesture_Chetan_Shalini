#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:13:36 2019

@author: shalini

"""

#------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a video sequence
#------------------------------------------------------------

# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import os
import glob
from sklearn.svm import SVC
import random
from train import generate_descriptor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

#-----------------
# TRAINING
#-----------------

# global variables
bg = None
cwd = os.getcwd()
#descriptors = np.random.normal(0.5, 1, size=(12, 12)) #Todo: Store real descriptors
labels = range(12)
sample_images = [cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm"]


def generate_features_and_labels(descriptors, labels):
    #creating labelEncoder
    le = preprocessing.OneHotEncoder()
    print("descriptors SHAPE", descriptors.shape)
    print("LABEL SHAPE", labels.shape)
    # Converting string labels into numbers.
    features = le.fit_transform(descriptors, labels)
    print(features[0]) 
    encoded_labels = le.fit_transform(labels)
    print("FEATURES SHAPE", features.shape)
    print("LABEL SHAPE", encoded_labels.shape)
    return features, encoded_labels

def knn_classification(descriptors, labels):
    #features, encoded_labels = generate_features_and_labels(descriptors, labels)
    model = KNeighborsClassifier(n_neighbors=3, weights="distance", p=2)
#    model = KNeighborsClassifier(n_neighbors=3, weights="distance", p=2)
    model.fit(descriptors,labels)
    return model

def svm_classification(descriptors, labels):
    #features, encoded_labels = generate_features_and_labels(descriptors, labels)
    model = SVC(gamma='auto')
    model.fit(descriptors, labels)
    return model

def prediction(model, test_descriptor):
    predicted= model.predict([test_descriptor])
    print(predicted)
    return predicted

#-----------------
# FUNCTIONS FOR TESTING AND ACCURACY
#-----------------

def map_sample_images(label, sample_images):
    return sample_images[label]


#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":

    # Load descriptors and labels for training
    descriptors = np.load('descriptor_training_data.npy')
    print("descriptors_shape", descriptors.shape)
    labels = np.load('descriptor_training_labels.npy')
    print("labels_shape", labels.shape)

    # Train and save data
    if os.path.exists("knn_training_weights.sav"):
      # Load knn model
      knn_model = pickle.load(open("knn_training_weights.sav", 'rb'))
    else:
      # Generate knn model
      knn_model = knn_classification(descriptors, labels)
      pickle.dump(knn_model, open("knn_training_weights.sav", 'wb'))
     
        
    #Please enter path of your own image folder here, with subfolders open_hand, thumbs_up, v and three.
    image_folder = "Custom_Test/test/"
    gesture_types = ["open_hand", "thumbs_up", "v", "three"]
    gesture_labels = [0, 1, 2, 3]
    i = 0
    for gesture_type in gesture_types:
        gesture_path = image_folder + gesture_type
        acc = 0
        count = 0
        for gesture_image in glob.glob(gesture_path + '/*.png'):
          distSig = generate_descriptor(gesture_image)
          pred = prediction(knn_model, distSig)
          if(pred == gesture_labels[i]):
              acc = acc + 1
          print("Prediction = ", pred)
          count = count + 1
        print("-------------------Image: ", gesture_image, "\nClass: ", gesture_types[i], " accuracy: ", ((acc / count) * 100), "%------------------------------")
        i = i + 1

#-----------------
