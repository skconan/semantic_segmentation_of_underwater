#!/usr/bin/env python
"""
    File name: semantic_segmentation_ae.py
    Author: skconan
    Date created: 2019/07/13
    Python Version: 3.6
    Source: Simple CNN, Autoencoder
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, LeakyReLU, SeparableConv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D, Dropout
import keras
from keras.callbacks import ModelCheckpoint

from utilities import *
import time

img_dir = "./dataset/images"
target_dir = "./dataset/groundTruth_color_train"
test_dir = "./dataset/groundTruth_color_test"

result_dir = "./cnn_ae_" + str(time.time()).split(".")[0]

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    os.mkdir(result_dir+"/model")
    os.mkdir(result_dir+"/predict_result")

TRAIN_IMAGES = []
TRAIN_TARGET_IMAGES = get_file_path(target_dir)
TEST_IMAGES = []
TEST_TARGET_IMAGES = get_file_path(test_dir)

img_col = 256
img_row = 256

for target_path in TRAIN_TARGET_IMAGES:
    name = get_file_name(target_path)
    img_path = img_dir + "/" + name + ".jpg"
    if not os.path.exists(img_path):
        continue
    TRAIN_IMAGES.append(img_path)


for test_path in get_file_path(test_dir):
    name = get_file_name(test_path)
    img_path = img_dir + "/" + name + ".jpg"
    if not os.path.exists(img_path):
        continue
    TEST_IMAGES.append(img_path)


def load_image(path):
    image_list = np.zeros((len(path),  img_row, img_col, 3))
    for i, fig in enumerate(path):
        try:
            img = image.load_img(fig, target_size=(img_row, img_col, 3))
        except:
            print("error")
        x = image.img_to_array(img).astype('float32')
        x = (x / 255.)
        # x = (x / 127.5)-1.
        image_list[i] = x

    return image_list


x_train = load_image(TRAIN_IMAGES)
y_train = load_image(TRAIN_TARGET_IMAGES)
x_test = load_image(TEST_IMAGES)
y_test = load_image(TEST_TARGET_IMAGES)

x_val, y_val = x_test, y_test

print(x_train.shape, x_test.shape)

class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
    
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print('Training: epoch {} ends at {}'.format(epoch, time.time() - self.start_time))
        if epoch % 20 == 0:
            model_path = result_dir + "/model"
            model_list = get_file_path(model_path)
            if len(model_list) > 5:
                model_list = sorted(model_list, reverse=True)
                for m_path in model_list[5:]:
                    os.remove(m_path)
                    print("remove " + m_path)
            preds = self.model.predict(x_test)
            for i in range(10):
                preds_0 = preds[i] * 255.
       
                preds_0 = np.uint8(preds_0.reshape(img_row, img_col, 3))
                preds_0 = cv.resize(preds_0.copy(), (484,304))
                x_test_0 = x_test[i] * 255.
         
                x_test_0 = np.uint8(x_test_0.reshape(img_row, img_col, 3))
                x_test_0 = cv.resize(x_test_0.copy(), (484,304))
                plt.imshow(x_test_0)
                plt.savefig(result_dir + "/predict_result/" + str(epoch) + "_" + str(i) + "_a _test.jpg")
                plt.close()
                plt.imshow(preds_0)
                plt.savefig(result_dir + "/predict_result/" + str(epoch) + "_" + str(i) + "_b_pred.jpg")
                plt.close()



class Autoencoder():
    def __init__(self):
        self.img_rows = img_row
        self.img_cols = img_col
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=sgd, metrics = ['acc'])
  
        self.autoencoder_model.summary()

    def build_model(self):
        input_layer = Input(shape=(img_row, img_col, 3))

        x = Conv2D(128, (3, 3), padding='same',
                   name='Down_Conv1_1')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)        
        
        x = MaxPooling2D((2, 2), name='pool1')(x)

        x = Conv2D(128, (3, 3),
                   padding='same', name='Down_Conv2_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)        
        
        x = MaxPooling2D((2, 2), name='pool2')(x)

        x = Conv2D(256, (3, 3),
                   padding='same', name='Down_Conv3_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)        
      
        x = MaxPooling2D((2, 2), name='pool3')(x)

        x = Conv2D(512, (3, 3),
                   padding='same', name='Down_Conv4_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)        
       
        
        x = Dropout(0.3)(x)

        x = MaxPooling2D((2, 2), name='pool4')(x)

        x = Conv2D(1024, (3, 3),
                   padding='same', name='Mid_Conv1_1')(x)
    
        x = BatchNormalization()(x)
        x = Activation('relu')(x)        

        x = Dropout(0.3)(x)
        
# ################################

        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='Up_Conv1_1')(x)
                


        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='Up_Conv2_1')(x)
            
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='Up_Conv3_1')(x)
                 
        x = UpSampling2D((2, 2))(x)


        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='Up_Conv4_1')(x)
           
        x = UpSampling2D((2, 2))(x)

        output_layer = Conv2D(
            3, (3, 3), activation='sigmoid', padding='same')(x)

        return Model(input_layer, output_layer)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')

        filepath = result_dir +  "/model/model-{epoch:02d}-{val_loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='min')
        my_callback = MyCallback()
        callbacks_list = [
            checkpoint,
            my_callback
        ]
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=callbacks_list
                                             )
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        # plt.savefig(result_dir + "/graph.jpg")
        # plt.close()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

    def save(self, path):
        self.autoencoder_model.save(path)



ae = Autoencoder()
ae.train_model(x_train, y_train, x_val, y_val, epochs=3000, batch_size=10)
