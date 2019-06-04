# Facial-Recognition-via-CNN
This is a facial recognition method by using Convolutional Neural Network. In this project we can distinguish between images by extracting features of images

For dataset we have to make a folder name dataset in the same folder where cnn code is kept.
In dataset folder we have to make two subfolder name = test_set and training_set .
test_set and training_set contains two classes which can be two species like cat and dog or in my case i have entered two person name ankit and ironman.In both test_set and training_set folder make two folder with name (cat and dog) or (ankit and ironman).
Under ankit folder keep all the images of ankit and in ironman folder keep all the images of ironman.
Images in test_set and training_set folder should be of ratio 20:80.For better result try to keep as much image as you can.

Here model_joblib is used as a brain of my cnn model which contain all the information of output of all the epoch which saves the time to run the model again and again. 
