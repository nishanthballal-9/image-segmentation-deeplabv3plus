import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
import tensorflow_addons as tfa

img_w, img_h = 256, 256
#batch_size = 16
num_classes = 3

def AtrousSpatialPyramidPooling(model_input):
    dims = tf.keras.backend.int_shape(model_input)

    layer = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3],
                                                      dims[-2]))(model_input)
    layer = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same',
                                 kernel_initializer = 'he_normal')(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    out_pool = tf.keras.layers.UpSampling2D(size = (dims[-3] // layer.shape[1],
                                               dims[-2] // layer.shape[2]),
                                        interpolation = 'bilinear')(layer)
  
    layer = tf.keras.layers.Conv2D(256, kernel_size = 1,
                                   dilation_rate = 1, padding = 'same',
                                   kernel_initializer = 'he_normal',
                                   use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_1 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                   dilation_rate = 6, padding = 'same', 
                                   kernel_initializer = 'he_normal',
                                   use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_6 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                   dilation_rate = 12, padding = 'same',
                                   kernel_initializer = 'he_normal',
                                   use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_12 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                   dilation_rate = 18, padding = 'same',
                                   kernel_initializer = 'he_normal',
                                   use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_18 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Concatenate(axis = -1)([out_pool, out_1,
                                                    out_6, out_12,
                                                    out_18])

    layer = tf.keras.layers.Conv2D(256, kernel_size = 1,
                                   dilation_rate = 1, padding = 'same',
                                   kernel_initializer = 'he_normal',
                                   use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    model_output = tf.keras.layers.ReLU()(layer)
    return model_output

def DeeplabV3Plus(nclasses = 3):
    model_input = tf.keras.Input(shape=(img_h,img_w,3))
    resnet50 = tf.keras.applications.ResNet50(weights = 'imagenet',
                                            include_top = False,
                                            input_tensor = model_input)
    layer = resnet50.get_layer('conv4_block6_2_relu').output
    layer = AtrousSpatialPyramidPooling(layer)
    input_a = tf.keras.layers.UpSampling2D(size = (img_h // 4 // layer.shape[1],
                                                 img_w // 4 // layer.shape[2]),
                                          interpolation = 'bilinear')(layer)

    input_b = resnet50.get_layer('conv2_block3_2_relu').output
    input_b = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same',
                                   kernel_initializer = tf.keras.initializers.he_normal(),
                                   use_bias = False)(input_b)
    input_b = tf.keras.layers.BatchNormalization()(input_b)
    input_b = tf.keras.layers.ReLU()(input_b)

    layer = tf.keras.layers.Concatenate(axis = -1)([input_a, input_b])

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                   padding = 'same', activation = 'relu',
                                   kernel_initializer = tf.keras.initializers.he_normal(),
                                   use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2D(256, kernel_size =3,
                                   padding = 'same', activation = 'relu',
                                   kernel_initializer = tf.keras.initializers.he_normal(),
                                   use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.UpSampling2D(size = (img_h // layer.shape[1],
                                                 img_w // layer.shape[2]),
                                          interpolation = 'bilinear')(layer)
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size = (1,1),
                                   padding = 'same')(layer)
    return tf.keras.Model(inputs = model_input, outputs = model_output)