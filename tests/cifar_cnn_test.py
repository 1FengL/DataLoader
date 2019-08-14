from dataloader.dataset import *
from dataloader.base import *
from dataloader.image import *
from dataloader.utils import *

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)
from tensorlayer.models import Model


RESIZE_HEIGHT = 32
RESIZE_WIDTH = 32


# define the network
def get_model(inputs_shape):
    # self defined initialization
    W_init = tl.initializers.truncated_normal(stddev=5e-2)
    W_init2 = tl.initializers.truncated_normal(stddev=0.04)
    b_init2 = tl.initializers.constant(value=0.1)

    # build network
    ni = Input(inputs_shape)
    nn = Conv2d(64, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv1')(ni)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)
    nn = LocalResponseNorm(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")(nn)

    nn = Conv2d(64, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv2')(nn)
    nn = LocalResponseNorm(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

    nn = Flatten(name='flatten')(nn)
    nn = Dense(384, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense1relu')(nn)
    nn = Dense(192, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense2relu')(nn)
    nn = Dense(10, act=None, W_init=W_init2, name='output')(nn)

    M = Model(inputs=ni, outputs=nn, name='cnn')
    return M


def measure_dl_speed(dl, num_steps):

    # get the network
    net = get_model([None, 32, 32, 3])
    net.train()

    train_weights = net.trainable_weights
    optimizer = tf.optimizers.Adam(learning_rate=1e-6)

    cnt = 0
    loading_time_sum, training_time_sum = 0, 0
    rss_sum, vms_sum, shared_sum = 0, 0, 0
    rss_before, vms_before, shared_before = get_process_memory()
    loading_time_start = time.time()
    epoch = 0
    while epoch >= 0:
        if cnt == num_steps:
            break
        epoch += 1
        print("epoch: ", epoch)
        for img, label in dl:
            loading_time_end = time.time()
            loading_time_sum += loading_time_end - loading_time_start
            rss_after, vms_after, shared_after = get_process_memory()
            rss_sum += rss_after - rss_before
            vms_sum += vms_after - vms_before
            shared_sum += shared_after - shared_before

            training_time_start = time.time()

            with tf.GradientTape() as tape:
                logits = net(img)
                loss = tl.cost.cross_entropy(output=logits, target=label)
                acc = np.mean(np.equal(np.argmax(logits, 1), label))

            grad = tape.gradient(loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

            training_time_end = time.time()
            training_time_sum += training_time_end - training_time_start

            print("Loss: ", loss.numpy(), "Acc: ", acc, " | Loading: ", loading_time_end - loading_time_start, " | Training: ",
                  training_time_end - training_time_start)

            cnt += 1
            if cnt == num_steps:
                break

            loading_time_start = time.time()

    print("Average loading: ", loading_time_sum / num_steps, " | Average training: ", training_time_sum / num_steps)
    print("Average RSS: ", format_bytes(rss_sum / num_steps), " | Average VMS: ", format_bytes(vms_sum / num_steps),
          " | Average SHR: ", format_bytes(shared_sum / num_steps))


class myTransform(Transform):

    def __call__(self, img, label):
        img = tl.prepro.imresize(img, (RESIZE_HEIGHT, RESIZE_WIDTH)).astype(np.float32) / 255
        return img, label


if __name__ == '__main__':
    ds = CIFAR10(train_or_test='train')
    dl = Dataloader(ds, output_types=(np.float32, np.int32), batch_size=512, shuffle=True, num_worker=1,
                    transforms=[myTransform()])
    print("######  Dataloader  ######")
    measure_dl_speed(dl, num_steps=1000)
