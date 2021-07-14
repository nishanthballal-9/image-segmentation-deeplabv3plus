import tensorflow as tf
import numpy as np
from model import *
import time
import os
import cv2
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import pickle
from model import DeeplabV3Plus

test_dir = "Dataset4/Val_Image_no_overlap"
model_path = "model/exp5_cce_loss_overlap_256/deeplab_model_best_val_mIoU.h5"
batch_size = 16
dest_path = 'Val_Results_exp5'

#Pixel value for BGR image
idx_to_class = {0:[0,0,0], 1:[0,0,255], 2:[0,255,255]}

#Custome Metric: Mean IoU
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
                y_true=None,
                y_pred=None,
                num_classes=None,
                name=None,
                dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true_f > 0.5, tf.uint8)
        y_pred = tf.cast(y_pred_f > 0.5, tf.uint8)
        return super().update_state(y_true, y_pred, sample_weight)

model = DeeplabV3Plus()
model.load_weights(model_path)

def preprocess_images(image_path):
    # Load and preprocess the image
    img = tf.io.read_file(image_path) # read the raw image
    img = tf.image.decode_png(img, channels=3) # decode the image back to proper format
    img.set_shape([None, None, 3])
    img = tf.cast(img,tf.float32) / 127.5 - 1 #Normalize

    return img

def mask_to_rgb(img, idx_to_class):
    """
        converts output sparse categorical mask to 3 channel rgb mask

        Parameters: img -> RGB segmentation mask
                    idx_to_class -> dictionary mapping class index to pixel
                                    for eg. {0: [255,255,255],
                                             1: [0,0,0]}
                                    can be generated like:
                                    idx_to_class = {i: pixel for i, pixel in enumerate(PIXELS)}
                                    where PIXELS is list of pixels like [np.array([0,0,0]),np.array(255,255,255)]
    
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    ret_img = np.zeros((img_height, img_width, 3))
    for i in range(img_height):
        for j in range(img_width):
            pixel = idx_to_class[img[i][j]]
            ret_img[i][j] = pixel

    print(ret_img.shape)
    return ret_img

image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
print(len(image_paths))
list_ds = tf.data.Dataset.from_tensor_slices((image_paths))
test_ds = list_ds.map(preprocess_images, num_parallel_calls=6)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

start = time.time()
results = model.predict(test_ds, steps = 175)

#Categorical encoding from softmax values
results = np.argmax(results, axis=3)

#Converting categorical encoding to bgr
results_rgb = [mask_to_rgb(results[i], idx_to_class) for i in range(results.shape[0])]
results_rgb = np.array(results_rgb)
end = time.time()

print(end-start)
print(results.shape)
print(results_rgb.shape)

#Saving Validation Patches
[cv2.imwrite(os.path.join(dest_path, image_paths[i].split('/')[-1]), results_rgb[i]) for i in range(len(results_rgb))]
