#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:38 2020

@author: jameselijah
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()


plt.imshow(training_images[2])
#plt.show()
#print(training_images[0]) #get the raw value inputs of the shoe

training_images = training_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([ #3 LAYERS
    keras.layers.Flatten(input_shape = (28,28)), #specify the shape and then Flatten makes it into a single dimensional array
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax) #10 neurons bc we have 10 classes of clothing in the dataset (must always match)
    ])

model.compile(optimizer = tf.train.AdamOptimizer(), 
              loss "sparse_categorical_crossentropy")

model.fit(training_images, training_labels, epochs = 5)

