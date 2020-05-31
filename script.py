import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from main_model import model, train

x = list()
y = list()

num_images = 400

image_dir = 'cityscape_images/images'
image_filenames = os.listdir(image_dir)
for filename in image_filenames[0: 100]:
    image = Image \
        .open(os.path.join(image_dir, filename))
    x.append(np.asarray(ImageOps.crop(image, (0, 0, 256, 0)).resize((128, 128))))
    y.append(np.asarray(ImageOps.crop(image, (256, 0, 0, 0)).resize((128, 128))))

x = np.array(x) / 255
y = np.array(y)

train_features, test_features, train_labels, test_labels = train_test_split(np.array(x), np.array(y),
                                                                            test_size=0.4)


def binarize(pixel):
    if np.array_equal(pixel, [128, 63, 127]):
        return np.array([1])
    else:
        return np.array([0])


train_labels = np.apply_along_axis(binarize, axis=3, arr=train_labels)
test_labels = np.apply_along_axis(binarize, axis=3, arr=test_labels)

# @markdown > The batch size for the dataset.
batch_size = 5  # @param {type: "number"}

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset = train_dataset.shuffle(1024).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
test_dataset = test_dataset.shuffle(1024).batch(batch_size)

import datetime

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

# @markdown > Number of epochs for training the model
num_epochs = 25  # @param {type: "number"}

for e in range(num_epochs):
    print('Epoch {} out of {} {}'.format(e + 1, num_epochs, '--' * 50))
    for features in train_dataset:
        image, label = features
        summ_loss = train(model, image, label)
        tf.summary.scalar('loss', data=summ_loss, step=e)

import matplotlib.pyplot as plt

input_image = test_features[0:1]
pred = model(input_image).numpy()
image = np.zeros((128, 128, 3))
for x in range(128):
    for y in range(128):
        if pred[0, x, y] > 0.5:
            image[x, y] = [255, 255, 255]
        else:
            image[x, y] = [0, 0, 0]


def show_images(images: list):
    n = len(images)
    f = plt.figure()
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i], interpolation='none')
    plt.show()


show_images([test_features[0], image])
