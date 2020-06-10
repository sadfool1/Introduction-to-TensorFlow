import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

"""
Next is to define your model. Now instead of the input layer at the top, you're going to add a Convolution. 
The parameters are:

1. The number of convolutions you want to generate. Purely arbitrary, but good to start with something in 
the order of 32

2. The size of the Convolution, in this case a 3x3 grid

3. The activation function to use -- in this case we'll use relu, which you might recall is the equivalent 
of returning x when x>0, else returning 0

4. In the first layer, the shape of the input data.

You'll follow the Convolution with a MaxPooling layer which is then designed to compress the image, while 
maintaining the content of the features that were highlighted by the convlution. By specifying (2,2) for 
the MaxPooling, the effect is to quarter the size of the image. Without going into too much detail here, 
the idea is that it creates a 2x2 array of pixels, and picks the biggest one, thus turning 4 pixels into 1.
It repeats this across the image, and in so doing halves the number of horizontal, and halves the number 
of vertical pixels, effectively reducing the image by 25%.

You can call model.summary() to see the size and shape of the network, and you'll notice that after every 
MaxPooling layer, the image size is reduced in this way.
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
  ])

#Now compile the model, call the fit method to do the training, and evaluate the loss and accuracy from the test set.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)