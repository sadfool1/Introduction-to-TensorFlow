#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:56:38 2020

@author: jameselijah
"""
"""
Beyond Hello World, A Computer Vision Example
In the previous exercise you saw how to create a neural network that figured out the problem you 
were trying to solve. This gave an explicit example of learned behavior. Of course, in that instance, 
it was a bit of overkill because it would have been easier to write the function Y=2x-1 directly, 
instead of bothering with using Machine Learning to learn the relationship between X and Y for a fixed 
set of values, and extending that for all values.


But what about a scenario where writing rules like that is much more difficult -- for example a 
computer vision problem? Let's take a look at a scenario where we can recognize different items of 
clothing, trained from a dataset containing 10 different types.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

np.set_printoptions(linewidth=200)

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.fit(training_images, training_labels, epochs = 5)

#model.evaluate(test_images, test_labels) #now evaluate unforeseen data

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#CALL BACK FUNC to stop when desired accurcy hits. In this case loss < 0.1 means we are looking for 90% accuracy.
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.1):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])


