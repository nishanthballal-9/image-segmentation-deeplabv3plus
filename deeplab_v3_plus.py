import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
import tensorflow_addons as tfa
from model import DeeplabV3Plus

img_w, img_h = 512, 512
batch_size = 16
num_classes = 3
log_addr = 'logs/'
SMOOTH = 1e-5

#Classes Black, Red and Yellow - RGB Conversions
colors = [(0,0,0), (255,0,0), (255,255,0)]

#Custom Metric Mean IoU
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

# Custom loss function: Dice Loss
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#Tversky Index
def tversky_index(y_true, y_pred):
    # generalization of dice coefficient algorithm
    #   alpha corresponds to emphasis on False Positives
    #   beta corresponds to emphasis on False Negatives (our focus)
    #   if alpha = beta = 0.5, then same as dice
    #   if alpha = beta = 1.0, then same as IoU/Jaccard
    alpha = 0.5
    beta = 0.5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection) / (intersection + alpha * (K.sum(y_pred_f*(1. - y_true_f))) + beta *  (K.sum((1-y_pred_f)*y_true_f)))

#Tversky Loss
def tversky_index_loss(y_true, y_pred):
    return -tversky_index(y_true, y_pred)

#Function to read image and mask: augmentation applied to original images and mask converted from rgb to one hot encoding
def read_train_img(image_path,mask=False, flip=0, rotate=0, color_aug=0, train=True, colors_list=[(0,0,0), (255,0,0), (255,255,0)]):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        one_hot_map = []
        for color in colors_list:
            class_map = tf.reduce_all(tf.equal(img,color), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        img = tf.cast(one_hot_map, tf.float32)
        img = tf.case([(tf.greater(rotate, 0), lambda: tf.image.rot90(img))], default=lambda: img)
        img = tf.case([(tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))], default=lambda: img)
        #img = (tf.image.resize(images=img, size=[
        #              img_h, img_w]))
        img = tf.cast(img,tf.float32)
    else:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        #img = tf.image.rot90(img)
        img = tf.case([(tf.greater(rotate, 0), lambda: tf.image.rot90(img))], default=lambda: img)
        #img = tf.image.random_brightness(img, max_delta=0.05)
        #img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        #img = tf.image.random_hue(img, max_delta=0.08)
        #img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.case([(tf.greater(color_aug, 0), lambda: tf.image.random_brightness(img, max_delta=0.05))], default=lambda: img)
        img = tf.case([(tf.greater(color_aug, 0), lambda: tf.image.random_saturation(img, lower=0.5, upper=1.5))], default=lambda: img)
        img = tf.case([(tf.greater(color_aug, 0), lambda: tf.image.random_hue(img, max_delta=0.08))], default=lambda: img)
        img = tf.case([(tf.greater(color_aug, 0), lambda: tf.image.random_contrast(img, lower=0.7, upper=1.3))], default=lambda: img)
        #img = tf.clip_by_value(img, 0, 255)
        #img = (tf.image.resize(images=img, size=[
        #              img_h, img_w]))
        
        img = tf.case([(tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))], default=lambda: img)
        #using mean and sd from train data
        #img = tf.cast(img,tf.float32) - 232.1207 / 40.1743
        img = tf.cast(img,tf.float32) / 127.5 - 1
    return img

def read_val_img(image_path,mask=False, flip=0, rotate=0, color_aug=0, train=True, colors_list=[(0,0,0), (255,0,0), (255,255,0)]):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        one_hot_map = []
        for color in colors_list:
            class_map = tf.reduce_all(tf.equal(img,color), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        img = tf.cast(one_hot_map, tf.float32)
        
        #img = tf.cast(img,tf.float32)
    else:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        #img = tf.cast(img,tf.float32) - 232.1207 / 40.1743    
        img = tf.cast(img,tf.float32) / 127.5 - 1
    return img

#preprocess function for tf.data pipeline
def load_train_data(img_list,mask_list):
    flip = tf.random.uniform(shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    rotate = tf.random.uniform(shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    color_aug = tf.random.uniform(shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    img = read_train_img(img_list, flip = flip, rotate=rotate, color_aug=color_aug)
    mask = read_train_img(mask_list,mask=True, flip = flip, rotate=rotate)
    #Code to generate random crop
    return img, mask

#preprocess function for tf.data pipeline
def load_val_data(img_list,mask_list):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    img = read_val_img(img_list, flip = flip)
    mask = read_val_img(mask_list,mask=True, flip=flip)
    #Code to generate random crop
    return img, mask

def data_generator(img_list,mask_list,batch_size,train=True):
    dataset = tf.data.Dataset.from_tensor_slices((img_list,
                                                    mask_list))
    if train:
        dataset = dataset.shuffle(buffer_size=128)
        dataset = dataset.repeat()
        dataset = dataset.map(load_train_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(load_val_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
    return dataset

train_images_folder = sorted(glob('Dataset4/Train_Image/*'))
train_mask_folder = sorted(glob('Dataset4/Train_Mask/*'))
valid_images_folder = sorted(glob('Dataset4/Val_Image_no_overlap/*'))
#valid_images_folder = sorted(glob('Datset4/Val_Image_no_overlap/*'))
valid_mask_folder = sorted(glob('Dataset4/Val_Mask_no_overlap/*'))
print(len(train_images_folder))
print(len(train_mask_folder))
print(len(valid_images_folder))
print(len(valid_mask_folder))

train_dataset = data_generator(train_images_folder,train_mask_folder,batch_size)
valid_dataset = data_generator(valid_images_folder,valid_mask_folder,batch_size,train=False)

model = DeeplabV3Plus()

model_path = "model/exp4_cce_loss_overlap_data/deeplab_model_best_val_mIoU.h5"

model.load_weights(model_path)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

metrics = ['accuracy', tfa.metrics.F1Score(num_classes = 3, threshold=0.1, average='micro', name='mean_dice'), UpdatedMeanIoU(num_classes=3, name='mean_iou', dtype='float32')]

model.summary()

epochs=300
step_per_epoch= 2*len(train_images_folder)//batch_size
val_step_per_epoch = len(valid_images_folder)//batch_size

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=loss,metrics=metrics)

filepath1 = 'model/deeplab_model_best_val_loss.h5'

checkpoint1 = ModelCheckpoint(filepath1, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              save_weights_only=False, 
                              mode='min')

filepath2 = 'model/deeplab_model_best_val_mDice.h5'

checkpoint2 = ModelCheckpoint(filepath2, 
                              monitor='val_mean_dice', 
                              save_best_only=True, 
                              save_weights_only=False, 
                              mode='max')

filepath3 = 'model/deeplab_model_best_val_mIoU.h5'

checkpoint3 = ModelCheckpoint(filepath3, 
                              monitor='val_mean_iou', 
                              save_best_only=True, 
                              save_weights_only=False, 
                              mode='max')

csv_logger = CSVLogger(log_addr+"/train.csv", append=True, separator=',')
early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose =1, mode="min")
cb = [checkpoint1, checkpoint2, checkpoint3, csv_logger, early_stopping, reduce_lr]

history = model.fit(train_dataset,steps_per_epoch=step_per_epoch, epochs=epochs,
          validation_data=valid_dataset,
          validation_steps=val_step_per_epoch, 
          callbacks=cb)

#loss values over training and plots.
loss = np.asarray(history.history['loss'])
val_loss = np.asarray(history.history['val_loss'])
mIOU = np.asarray(history.history['mean_iou'])
val_mIOU = np.asarray(history.history['val_mean_iou'])
mDice = np.asarray(history.history['mean_dice'])
val_mDice = np.asarray(history.history['val_mean_dice'])
e= range(len(loss))
#plot
plt.plot(e, loss, 'r-', label= 'training loss')
plt.plot(e, val_loss, 'b-', label = 'validation loss')
plt.xlabel('Epochs')
plt.ylabel('bce_dice_loss (train/val)')
plt.savefig(log_addr+'loss_plot.png')
plt.close()

plt.plot(e, mDice, 'g-', label = 'Training acc')
plt.plot(e, val_mDice, 'c-', label = 'Validation acc')
plt.xlabel('Epochs')
plt.ylabel('mean_dice')
plt.savefig(log_addr+'metric.png')
plt.close()

#saving the stats
np.savetxt(log_addr+'training_loss.txt', loss)
np.savetxt(log_addr+'training_val_loss.txt', val_loss)
np.savetxt(log_addr+'training_metric_miou.txt', mIOU)
np.savetxt(log_addr+'validation_metric_miou.txt', val_mIOU)
np.savetxt(log_addr+'training_metric_mDice.txt', mDice)
np.savetxt(log_addr+'validation_metric_mDice.txt', val_mDice)
