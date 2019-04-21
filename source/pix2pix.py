from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from utilities import *
import cv2 as cv


def load( img_file, label_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img)
    in_img = img

    img = tf.io.read_file(label_file)
    img = tf.image.decode_jpeg(img)
    out_img = img

    in_img = tf.cast(in_img, tf.float32)
    out_img = tf.cast(out_img, tf.float32)

    return in_img, out_img


def resize(in_img, out_img, height, width):
    in_img = tf.image.resize(in_img, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    out_img = tf.image.resize(out_img, [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # in_img = cv.resize(in_img, (height, width))
    # out_img = cv.resize(out_img, (height, width))
    return in_img, out_img


def random_crop(in_img, out_img):
    if (out_img.shape[2] == 1):
        out_img = tf.image.grayscale_to_rgb(out_img)
    # print(in_img.shape, out_img.shape)
    stacked_image = tf.stack([in_img, out_img], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]
    # return in_img, out_img


def normalize(in_img, out_img):
    # normalizing the images to [-1, 1]

    in_img = (in_img / 127.5) - 1
    out_img = (out_img / 127.5) - 1

    return in_img, out_img


def random_jitter( in_img, out_img):
    # resizing to 286 x 286 x 3
    in_img, out_img = resize(in_img, out_img, 286, 286)

    # randomly cropping to 256 x 256 x 3
    in_img, out_img = random_crop(in_img, out_img)

    val = tf.random.uniform(())
    if val > 0.5:
        # random mirroring
        in_img = tf.image.flip_left_right(in_img)
        out_img = tf.image.flip_left_right(out_img)

    return in_img, out_img


def load_image_train( img_file, label_file):
    in_img, out_img = load( img_file, label_file)
    in_img, out_img = random_jitter( in_img, out_img)
    # in_img, out_img = resize(in_img, out_img, 256, 256)

    in_img, out_img = normalize(in_img, out_img)

    return in_img, out_img


def load_image_test(img_file, label_file):
    in_img, out_img = load(img_file, label_file)
    in_img, out_img = resize(in_img, out_img,
                             IMG_HEIGHT, IMG_WIDTH)
    in_img, out_img = normalize(in_img, out_img)

    return in_img, out_img


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
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


def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs
    print(x)

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
    # print(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

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


def discriminator_loss(disc_real_output, disc_generated_output):
    # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = tf.keras.backend.binary_crossentropy(
        tf.ones_like(disc_real_output), disc_real_output, from_logits=True)

    generated_loss = tf.keras.backend.binary_crossentropy(tf.zeros_like(
        disc_generated_output), disc_generated_output, from_logits=True)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.keras.backend.binary_crossentropy(tf.ones_like(
        disc_generated_output), disc_generated_output, from_logits=True)
    # gan_loss = loss_object(tf.ones_like(
    #     disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


def generate_images(model, test_input, tar, epoch, j):
    print("Generate Image")
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input[tf.newaxis, ...], training=True)
    # plt.figure(figsize=(15, 15))
    # print(test_input.shape)
    display_list = [test_input, tar, prediction[0]]
    title = ['Input_Image', 'Ground_Truth', 'Predicted_Image']

    for i in range(3):
        plt.figure()
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig("train_pix2pix/" + str(epoch) + "_" + str(j) + "_" + title[i] + ".jpg")


def train_step(input_image, target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # gen_output = generator(input_image, training=True)
        gen_output = generator(input_image[tf.newaxis, ...], training=True)

        disc_real_output = discriminator(
            [input_image[tf.newaxis, ...], target[tf.newaxis, ...]], training=True)
        disc_generated_output = discriminator(
            [input_image[tf.newaxis, ...], gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 200
LAMBDA = 100
OUTPUT_CHANNELS = 3

generator_optimizer = None
discriminator_optimizer = None
discriminator = None
checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = None
saver = None
generator = None
global_step = None


def train( dataset, test_dataset, epochs):
    if not os.path.exists("./train_pix2pix"):
        os.mkdir("./train_pix2pix")
    for epoch in range(epochs):
        start = time.time()
        for input_image, target in dataset:
            train_step(input_image, target)
        clear_output(wait=True)
        i = 0
        for inp, tar in test_dataset[:10]:
            generate_images(generator, inp, tar, epoch, i)
            i += 1
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            # saver.save( checkpoint_prefix, global_step=epoch+1)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))
        # sess.run(tf.assign(global_step, epoch + 1))


def main():
    global generator, saver, global_step, checkpoint, generator_optimizer,discriminator_optimizer,discriminator
    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator = Discriminator()
    # saver = tf.train.Saver(max_to_keep=6)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

    train_dataset = []
    test_dataset = []
    img_dir = "./dataset_new/images_train"
    img_test_dir = "./dataset_new/images_test"

    label_dir = "./dataset_new/groundTruth_kmean_train"
    label_test_dir = "./dataset_new/groundTruth_kmean_test"

    # init = (tf.global_variables_initializer(),
    #         tf.local_variables_initializer())

    # with tf.Session() as sess:
        # print("Load dataset...")
        # sess.run(init)
    img_path_list = get_file_path(img_dir)
    for img_path in img_path_list:
        name = get_file_name(img_path)
        label_path = label_dir + "/" + name + ".jpg"
        if not os.path.exists(label_path):
            continue
        # img = cv.imread(img_path, 1)
        # label = cv.imread(label_path, 1)
        dataset = load_image_train( img_path, label_path)
        # print(len(dataset), dataset[0].shape, dataset[1].shape)
        train_dataset.append(dataset)
    img_path_list = get_file_path(img_test_dir)
    for img_path in img_path_list:
        name = get_file_name(img_path)
        label_path = label_test_dir + "/" + name + ".jpg"
        if not os.path.exists(label_path):
            continue
        # img = cv.imread(img_path, 1)
        # label = cv.imread(label_path, 1)
        dataset = load_image_train( img_path, label_path)
        # print(len(dataset), dataset[0].shape, dataset[1].shape)
        test_dataset.append(dataset)
    print("Number of training set:", len(train_dataset))
    print("Number of testing set:", len(test_dataset))
    print("Done")
    with tf.device('/GPU:0'):
        train( train_dataset, test_dataset, EPOCHS)
    # train( train_dataset, test_dataset, EPOCHS)
        
if __name__ == "__main__":
    main()
