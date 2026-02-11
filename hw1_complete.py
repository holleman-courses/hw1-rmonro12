#!/usr/bin/env python

# this is a test i have changed the file
# TensorFlow and tf.keras
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
  model = None # Add code to define model 1.
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

  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
