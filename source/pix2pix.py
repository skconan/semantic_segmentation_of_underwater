#!/usr/bin/env python
"""
    File name: pix2pix.py
    Author: skconan
    Date created: 2019/04/13
    Python Version: 3.6
    Source: https://www.tensorflow.org/alpha/tutorials/generative/pix2pix
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import cv2 as cv
import numpy as np
import tensorflow as tf
from utilities import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image


class Pix2Pix():
    def __init__(self, postfix, train=True):
        self.BUFFER_SIZE = 400
        self.BATCH_SIZE = 1
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.LAMBDA = 100
        self.OUTPUT_CHANNELS = 3

        if train:
            self.checkpoint_dir = './pix2pix_checkpoints' + postfix
            self.train_result = "./pix2pix_train_result" + postfix
            self.predict_result = "./pix2pix_predict_result" + postfix

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            if not os.path.exists(self.train_result):
                os.makedirs(self.train_result)

            if not os.path.exists(self.predict_result):
                os.makedirs(self.predict_result)

        self.generator = self.Generator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator = self.Discriminator()

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator, discriminator=self.discriminator)

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        layer = tf.keras.Sequential()
        layer.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            layer.add(tf.keras.layers.BatchNormalization())

        layer.add(tf.keras.layers.LeakyReLU())

        return layer

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        kernel_size = 3
        down_stack = [
            # (bs, 128, 128, 64)
            self.downsample(64, kernel_size, apply_batchnorm=False),
            self.downsample(128, kernel_size),  # (bs, 64, 64, 128)
            self.downsample(256, kernel_size),  # (bs, 32, 32, 256)
            self.downsample(512, kernel_size),  # (bs, 16, 16, 512)
            self.downsample(512, kernel_size),  # (bs, 8, 8, 512)
            self.downsample(512, kernel_size),  # (bs, 4, 4, 512)
            self.downsample(512, kernel_size),  # (bs, 2, 2, 512)
            self.downsample(512, kernel_size),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            # (bs, 2, 2, 1024)
            self.upsample(512, kernel_size, apply_dropout=True),
            # (bs, 4, 4, 1024)
            self.upsample(512, kernel_size, apply_dropout=True),
            # (bs, 8, 8, 1024)
            self.upsample(512, kernel_size, apply_dropout=True),
            self.upsample(512, kernel_size),  # (bs, 16, 16, 1024)
            self.upsample(256, kernel_size),  # (bs, 32, 32, 512)
            self.upsample(128, kernel_size),  # (bs, 64, 64, 256)
            self.upsample(64, kernel_size),  # (bs, 128, 128, 128)
        ]
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, kernel_size,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (bs, 256, 256, 3)

        concat = tf.keras.layers.Concatenate()
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

        x = tf.keras.layers.concatenate(
            [inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = tf.keras.backend.binary_crossentropy(
            tf.ones_like(disc_real_output), disc_real_output, from_logits=True)

        generated_loss = tf.keras.backend.binary_crossentropy(tf.zeros_like(
            disc_generated_output), disc_generated_output, from_logits=True)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = tf.keras.backend.binary_crossentropy(tf.ones_like(
            disc_generated_output), disc_generated_output, from_logits=True)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss

    def generate_images(self, model, test_input, tar, epoch, j):
        prediction = model(tf.expand_dims(test_input, 0), training=True)
        test_input = resize(test_input, 304, 484)
        tar = resize(tar, 304, 484)
        prediction = resize(prediction[0], 304, 484)

        display_list = [test_input, tar, prediction]
        title = ['Input_Image', 'Ground_Truth', 'Predicted_Image']

        for i in range(3):
            plt.figure()
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.savefig(self.train_result + "/" + str(epoch) + "_" +
                        str(j) + "_" + title[i] + ".jpg")
            plt.close()

    def predict_images(self, img):
        print("Predict Image")
        prediction = self.generator(tf.expand_dims(img, 0), training=True)
        result = (prediction[0] * 0.5 + 0.5)*255
        result = np.uint8(result)
        result = cv.resize(result, (484, 304))
        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)

        img = (img * 0.5 + 0.5)*255
        img = np.uint8(img)
        img = cv.resize(img, (484, 304))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        return img, result

    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(
                tf.expand_dims(input_image, 0), training=True)
            disc_real_output = self.discriminator(
                [tf.expand_dims(input_image, 0), tf.expand_dims(target, 0)], training=True)
            disc_generated_output = self.discriminator(
                [tf.expand_dims(input_image, 0), gen_output], training=True)

            gen_loss = self.generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        return gen_loss.numpy().mean(), disc_loss.numpy().mean()

    def train(self, dataset, test_dataset, epochs, file, restore=False):
        start_epoch = 1
        if restore:
            m = tf.train.latest_checkpoint(self.checkpoint_dir)
            print("Restore from:", m)
            self.checkpoint.restore(m)
            start_epoch = int(self.checkpoint.step)

        for epoch in range(start_epoch, epochs):
            start = time.time()
            self.checkpoint.step.assign_add(1)
            gen_loss_list = []
            disc_loss_list = []
            for input_image, target in dataset:
                gen_loss, disc_loss = self.train_step(input_image, target)
                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            clear_output(wait=True)

            checkpoint_prefix = os.path.join(
                self.checkpoint_dir, str(epoch) + "-ckpt")

            if epoch % 10 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)
                print('Model saved in epoch', epoch)
                i = 0
                for inp, tar in test_dataset[:10]:
                    self.generate_images(self.generator, inp, tar, epoch, i)
                    i += 1

            g_loss = np.mean(gen_loss_list)
            d_loss = np.mean(disc_loss_list)

            file.write(str(epoch)+",%.4f" %
                       (g_loss)+","+"%.4f" % (d_loss)+"\n")

            print("Time for epoch", epoch, "is", "%.2f" %
                  (time.time()-start), "G loss:", g_loss, "D loss:", d_loss)

    def predict(self, predict_dataset, model_file):
        self.checkpoint.restore(model_file)
        count = 0
        for img in predict_dataset:
            img, result = self.predict_images(img)
            cv.imwrite(self.predict_result + "/" + "%03d" %
                       (count) + "_segmentation.jpg", result)
            cv.imwrite(self.predict_result + "/" + "%03d" %
                       (count) + "_original.jpg", img)
            count += 1


def load_training_dataset():
    train_dataset = []

    img_dir = "./dataset/images"
    label_dir = "./dataset/groundTruth_obj_color"

    label_path_list = get_file_path(label_dir)
    for label_path in label_path_list:
        name = get_file_name(label_path)
        img_path = img_dir + "/" + name + ".jpg"

        if not os.path.exists(img_path):
            continue
        try:
            dataset = load_image_train(img_path, label_path)
        except:
            print("error load image")
        degrees = int(tf.random.uniform((), maxval=0.7)*10)

        val = tf.random.uniform(())
        if val > 0.5:
            img_in = rotation(dataset[0], degrees)
            img_out = rotation(dataset[1], degrees)
        else:
            img_in = rotation(dataset[0], -degrees)
            img_out = rotation(dataset[1], -degrees)

        dataset = (img_in, img_out)
        train_dataset.append(dataset)

    print("Number of training set:", len(train_dataset))
    return train_dataset


def load_testing_dataset():
    test_dataset = []

    img_dir = "./dataset/images"
    label_dir = "./dataset/groundTruth_color_test"

    label_path_list = get_file_path(label_dir)
    for label_path in label_path_list:
        name = get_file_name(label_path)
        img_path = img_dir + "/" + name + ".jpg"
        if not os.path.exists(img_path):
            continue
        dataset = load_image_test(img_path, label_path)
        test_dataset.append(dataset)

    print("Number of testing set:", len(test_dataset))
    return test_dataset


def load_predict_dataset(img_predict_dir):
    predict_dataset = []

    img_path_list = get_file_path(img_predict_dir)
    for img_path in img_path_list:
        dataset = load_image_predict(img_path)
        predict_dataset.append(dataset)

    print("Number of prediction set:", len(predict_dataset))
    return predict_dataset


def main():
    tf.config.set_soft_device_placement(True)
    # is_train = False
    is_train = True
    postfix = '_obj_color'
    pix2pix = Pix2Pix(postfix=postfix)

    if is_train:
        train_dataset = load_training_dataset()
        test_dataset = load_testing_dataset()
        with tf.device('/device:GPU:0'):
            a = input("Do you want to restore model? 1 Yes 2 No")
            if int(a) == 2:
                is_restore = False
            else:
                is_restore = True
            f = open("./loss"+postfix+".csv", "w+")

            pix2pix.train(train_dataset, test_dataset, 200, f, is_restore)
            f.close()
    else:
        model_file = "./pix2pix_checkpoints_color/50-ckpt-5"
        predict_dataset = load_predict_dataset(
            img_predict_dir="./dataset/from_web/images")
        pix2pix.predict(predict_dataset, model_file)


if __name__ == "__main__":
    main()
