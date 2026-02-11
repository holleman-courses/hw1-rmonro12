#!/usr/bin/env python

# TensorFlow and tf.keras
from xml.parsers.expat import model
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = tf.keras.Sequential ([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(10)
  ]) # Add code to define model 1.
  return model

def build_model2():
  model = None # Add code to define model 1.
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  # Now separate out a validation set.
  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)
  # choose num_val_samples indices up to the size of train_images, !replace => no repeats
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]

  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]
  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  train_hist = model1.fit(train_images, train_labels,
    validation_data=(val_images, val_labels), # or use `validation_split=0.1`
    epochs=30)
  model1.save('saved_models/volume_conv1')
  
  test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  model1.summary()

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
