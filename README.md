# Facial-Recognition-via-CNN
This is a facial recognition method by using Convolutional Neural Network. In this project we can distinguish between images by extracting features of images

Face recognition is the task of identifying an already detected object as a known or unknown face . Often the problem of face recognition is confused with the problem of face detection Face Recognition on the other hand is to decide if the "face" is someone known, or unknown, using for this purpose a database of faces in order to validate this input face.

Recognition algorithms can be divided into two main approaches:
Geometric: Is based on  geometrical  relationship  between  facial  landmarks,  or  in  other words the spatial configuration of facial features.
Photometric stereo: Used to  recover  the  shape  of  an  object  from  a  number  of images taken under different lighting conditions.

Here are the three elements that enter into the convolution operation:
•	Input image
•	Feature detector
•	Feature map

What you do is detect certain features, say, their eyes and their nose, for instance, and you immediately know who you are looking at.
These are the most revealing features, and that is all your brain needs to see in order to make its conclusion. Even these features are seen broadly and not down to their minutiae.
If your brain actually had to process every bit of data that enters through your senses at any given moment, you would first be unable to take any actions, and soon you would have a mental breakdown. Broad categorization happens to be more practical.
Convolutional neural networks operate in exactly the same way.


Note:-

For dataset we have to make a folder name dataset in the same folder where cnn code is kept.
In dataset folder we have to make two subfolder name = test_set and training_set .
test_set and training_set contains two classes which can be two species like cat and dog or in my case i have entered two person name ankit and ironman.In both test_set and training_set folder make two folder with name (cat and dog) or (ankit and ironman).
Under ankit folder keep all the images of ankit and in ironman folder keep all the images of ironman.
Images in test_set and training_set folder should be of ratio 20:80.For better result try to keep as much image as you can.

Here model_joblib is used as a brain of my cnn model which contain all the information of output of all the epoch which saves the time to run the model again and again. 
