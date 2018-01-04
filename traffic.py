import os
import skimage
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

from keras.models import load_model

import h5py

train_data_dir = 'F:\Machine Learning\TrafficSigns\Training';

def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

images, labels = load_data(train_data_dir)

def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

images64 = [skimage.transform.resize(image, (64,64)) for image in images]

X_train = np.array(images64)
y_train = np.array(labels)

y_train = to_categorical(y_train)


model = Sequential()

model.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=(64, 64, 3)))
model.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu'))
model.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(4, 4),
                 activation='relu'))
model.add(Conv2D(64, kernel_size=(4, 4),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(62, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=100)

model.save('traffic.h5')
