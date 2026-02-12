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

def build_model1():
  model = tf.keras.Sequential ([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(10)
  ])
  return model

def build_model2():
  model = tf.keras.Sequential ([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model

def build_model3():
  model = tf.keras.Sequential ([
    Input(shape=input_shape),
    layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model

def build_model50k():
  model = tf.keras.Sequential ([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10)
  ])
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':
  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]
  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]
  input_shape  = train_images.shape[1:]
  ########################################
  
  ### Build, compile, and train model 1
  model1 = build_model1()
  model1.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  
  train_hist = model1.fit(train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30)
  model1.save('saved_models/model_1_fcnn')
  test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  model1.summary()

  ### Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  
  train_hist = model2.fit(train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30)
  model2.save('saved_models/model_2_conv')
  test_loss, test_acc = model2.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  model2.summary()
  
  test_img = np.array(keras.utils.load_img(
      './hw1-rmonro12/test_image_bird.jpg',
      grayscale=False,
      color_mode='rgb',
      target_size=(32,32)))
  loaded_model = tf.keras.models.load_model('saved_models/model_2_conv')
  pred = loaded_model.predict(test_img.reshape(1,32,32,3))
  print(pred)

  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()
  model3.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  
  train_hist = model3.fit(train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30)
  model3.save('saved_models/model_3_separable_conv')
  test_loss, test_acc = model3.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  model3.summary()

  ### Build, compile, and train best model under 50k params
  model50k = build_model50k()
  model50k.summary()
  
  model50k.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  train_hist = model50k.fit(train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30)
  model50k.save('saved_models/best_model.h5')
  test_loss, test_acc = model50k.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)