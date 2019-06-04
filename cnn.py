# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 164,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 40)

#start importing the brain of the CNN

from sklearn.externals import joblib
joblib.dump(classifier,'model_joblib')
mj = joblib.load('model_joblib')


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/dogs/2017-07-13-23-03-24-501.jpg', target_size = (64, 64))

test_image = image.load_img('dataset/test_set/cats/Screenshot (339).png', target_size = (64, 64))

test_image
# This is test img
# first arg is the path
# img is 64x64 dims this is what v hv used in training so wee need to use exactly the same dims
# here also

test_image = image.img_to_array(test_image)
# Also in our first layer below it is a 3D array
# Step 1 - Convolution
# classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# this will convert from a 3D img to 3D array
test_image # shld gv us (64,64,3)

test_image = np.expand_dims(test_image, axis = 0)
# axis specifies the position of indx of the dimnsn v r addng
# v need to add the dim in the first position
test_image # now it shld show (1,64,64,3)



result = classifier.predict(test_image)

#this is for joblib purpose
result = mj.predict(test_image)

# v r trying to predict
result # gv us 1

print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'Ankit'
else:
    prediction = 'Ironman'

print(prediction)


'''#classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''