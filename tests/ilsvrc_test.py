import time

import tensorflow as tf
import cv2

from dataloader.dataset import *
from dataloader.base import *
from dataloader.image import *

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


def measure_dl_speed(dl, num_steps):
    net = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                  weights='imagenet',
                                                  classes=1000)
    weights = net.trainable_weights
    optimizer = tf.optimizers.Adam(1e-3)

    cnt = 0
    loading_sum = 0
    training_sum = 0
    loading_start = time.time()
    for img, label in dl:
        loading_end = time.time()
        loading_sum += loading_end - loading_start

        label = np.expand_dims(label, 1)
        label = np.array(label, dtype=np.float32)

        training_start = time.time()

        with tf.GradientTape() as tape:
            logits = net(img)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, logits))
        grad = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grad, weights))

        training_end = time.time()
        training_sum += training_start - training_end

        print("Loss: ", loss, " | Loading: ", loading_end - loading_start, " | Training: ",
              training_end - training_start)

        cnt += 1
        if cnt == num_steps:
            break

        loading_start = time.time()

    print("Average loading: ", loading_sum / num_steps, " | Average training: ", training_sum / num_steps)


class myTransform(Transform):

    def __call__(self, img, label):
        img = cv2.resize(img, (RESIZE_HEIGHT, RESIZE_WIDTH))
        return img, label


if __name__ == '__main__':
    ds = ILSVRC12(path='../data', train_or_test='train', meta_dir='../data')
    for img, label in ds:
        print(img.shape)
        break
    dl = Dataloader(ds, batch_size=500, shuffle=True, num_worker=4, transforms=[myTransform()])
    for img, label in dl:
        print(img.shape, label.shape)
        break
    measure_dl_speed(dl, num_steps=50)
