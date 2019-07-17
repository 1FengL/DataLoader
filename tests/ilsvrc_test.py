import tensorflow as tf

from dataloader.dataset import *
from dataloader.base import *
from dataloader.image import *


def measure_dl_speed(dl):
    net = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                  weights='imagenet',
                                                  classes=1000)
    weights = net.trainable_weights
    optimizer = tf.optimizers.Adam()
    for img, label in dl:
        label = np.expand_dims(label, 1)
        with tf.GradientTape() as tape:
            logits = net(img)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, logits))
        print(loss)
        grad = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grad, weights))


# class myTransform(Transform):
#
#     def __call__(self, img, label):
#         return gray2rgb(img), label


if __name__ == '__main__':
    # ds = MNIST(train_or_test='train', path='../data')
    ds = ILSVRC12(path='../data', train_or_test='train', meta_dir='../data')
    for img, label in ds:
        print(img.shape)
        break
    dl = Dataloader(ds, batch_size=128, shuffle=True, num_worker=4, transforms=None)
    for img, label in dl:
        print(img.shape, label.shape)
        break
    measure_dl_speed(dl)
