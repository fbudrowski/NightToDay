import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, Conv2DTranspose, Input
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from random import randint, random
from numpy import ones, zeros, asarray
from read_input import Generator, evaluate_photo, sony_dataset
import time
from os import makedirs, path
import glob
import matplotlib as plt
from datetime import datetime

model_save_path = '/home/franco/datasets/visualn/Fuji/saved_models/1581150994.4581282'

epoch = 28

g_model_AtoB = tf.keras.models.load_model(path.join(model_save_path, 'g_model_AtoB_{}'.format(epoch) + '.h5'))


photo_path = '/home/franco/datasets/visualn/Sony/short/00056_00_0.1s.ARW'
evaluate_photo(photo_path, 'generated_photo/' + str(time.time()) , g_model_AtoB)
