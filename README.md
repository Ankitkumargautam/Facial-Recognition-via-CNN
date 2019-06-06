# Facial-Recognition-via-CNN
This is a facial recognition method by using Convolutional Neural Network.In this project we can distinguish between images by extracting features of images.Similarly like we recognize any people on one site just by looking their feature like eyes,nose,hair etc rather than noticing full body .

It identifies the classes of images that we have provided for training and learn all the feature pixel information of the images and then used to classify the classes  of images and then for test image it will show the probability of that image belonging to which class. 

Often the problem of face recognition is confused with the problem of face detection Face Recognition on the other hand is to decide if the "face" is someone known, or unknown.

Note:-

For dataset we have to make a folder name dataset in the same folder where cnn code is kept.
In dataset folder we have to make two subfolder name = test_set and training_set .
test_set and training_set contains two classes which can be two species like cat and dog or in my case i have entered two person name ankit and ironman.In both test_set and training_set folder make two folder with name (cat and dog) or (ankit and ironman).
Under ankit folder keep all the images of ankit and in ironman folder keep all the images of ironman.
Images in test_set and training_set folder should be of ratio 20:80.For better result try to keep as much image as you can.

Here model_joblib is used as a brain of my cnn model which contain all the information of output of all the epoch which saves the time to run the model again and again. 
