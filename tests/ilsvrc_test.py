import tensorflow as tf
from torch.utils.data import DataLoader as torchDataloader

from dataloader.common import Dataloader, TFDataloader
from dataloader.dataset import *
from dataloader.image import *
from dataloader.utils import *

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


def measure_dl_speed(dl_choice, num_steps, batch_size, num_worker, warm_up=5):
    ds = ILSVRC12(path='/home/dsimsc/data/luoyifeng/ILSVRC12', train_or_test_or_val='train',
                  meta_dir='/home/dsimsc/data/luoyifeng/ILSVRC12', shape=(RESIZE_HEIGHT, RESIZE_WIDTH))
    assert dl_choice in ('tf', 'tl', 'torch')
    if dl_choice == 'tf':
        dl = TFDataloader(ds, output_types=(tf.float32, tf.int32), shuffle=False, batch_size=batch_size,
                          transforms=[myTransform()])
    elif dl_choice == 'tl':
        dl = Dataloader(ds, output_types=(np.float32, np.int32), batch_size=batch_size, shuffle=False,
                        num_worker=num_worker,
                        transforms=[myTransform()])
    elif dl_choice == 'torch':
        dl = torchDataloader(ds, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    net = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                  weights='imagenet',
                                                  classes=1000)
    weights = net.trainable_weights
    optimizer = tf.optimizers.Adam(1e-3)

    cnt = 0
    loading_time_sum, training_time_sum = 0, 0

    warm_up_cnt = 0
    loading_time_start = time.time()
    for img, label in dl:
        if warm_up_cnt < warm_up:
            pass
        else:
            loading_time_end = time.time()
            loading_time_sum += loading_time_end - loading_time_start

        if dl_choice == 'torch':
            img = img.numpy()
            label = label.numpy()

        training_time_start = time.time()

        with tf.GradientTape() as tape:
            logits = net(img)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
        grad = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grad, weights))

        if warm_up_cnt < warm_up:
            warm_up_cnt += 1
            print("warming up...")
        else:
            training_time_end = time.time()
            training_time_sum += training_time_end - training_time_start
            print("Loss: ", loss, " | Loading: ", loading_time_end - loading_time_start, " | Training: ",
                  training_time_end - training_time_start)
            cnt += 1
            if cnt == num_steps:
                break

        loading_time_start = time.time()

    print("Average loading: ", loading_time_sum / num_steps, " | Average training: ", training_time_sum / num_steps)


class myTransform(Transform):

    def __call__(self, img, label):
        img = img[:, ::-1, :]
        img -= 127.5
        img /= 127.5
        return img, label


if __name__ == '__main__':
    measure_dl_speed(dl_choice='tf', num_steps=10, num_worker=4, batch_size=32)
