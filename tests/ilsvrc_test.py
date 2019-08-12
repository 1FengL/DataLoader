import tensorflow as tf

from dataloader.common import Dataloader, TFDataloader
from dataloader.dataset import *
from dataloader.image import *
from dataloader.utils import *

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


def measure_dl_speed(dl, num_steps):
    net = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                  weights='imagenet',
                                                  classes=1000)
    weights = net.trainable_weights
    optimizer = tf.optimizers.Adam(1e-3)

    cnt = 0
    loading_time_sum, training_time_sum = 0, 0
    rss_sum, vms_sum, shared_sum = 0, 0, 0
    rss_before, vms_before, shared_before = get_process_memory()
    loading_time_start = time.time()
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
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
        grad = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grad, weights))

        training_time_end = time.time()
        training_time_sum += training_time_end - training_time_start

        print("Loss: ", loss, " | Loading: ", loading_time_end - loading_time_start, " | Training: ",
              training_time_end - training_time_start)

        cnt += 1
        if cnt == num_steps:
            break

        loading_time_start = time.time()

    print("Average loading: ", loading_time_sum / num_steps, " | Average training: ", training_time_sum / num_steps)
    print("Average RSS: ", format_bytes(rss_sum / num_steps), " | Average VMS: ", format_bytes(vms_sum / num_steps),
          " | Average SHR: ", format_bytes(shared_sum / num_steps))


class myTransform(Transform):

    def __init__(self, is_tf=True):
        self.is_tf = is_tf

    def __call__(self, img, label):
        if not self.is_tf:
            img = np.array(img, dtype=np.float32)
        img /= 255
        return img, label


if __name__ == '__main__':
    ds = ILSVRC12(path='/home/dsimsc/data/luoyifeng/ILSVRC12', train_or_test='train',
                  meta_dir='/home/dsimsc/data/luoyifeng/ILSVRC12', shape=(RESIZE_HEIGHT, RESIZE_WIDTH))
    for img, label in ds:
        print(img.shape, label)
        exit(1)
    dl = Dataloader(ds, output_types=(np.float32, np.int32), batch_size=32, shuffle=False, num_worker=4,
                    transforms=[myTransform()])
    print("######  Dataloader  ######")
    measure_dl_speed(dl, num_steps=10)

    dl = TFDataloader(ds, output_types=(tf.float32, tf.int32), shuffle=False, batch_size=32,
                      transforms=[myTransform()])
    print("######  Dataloader  ######")
    measure_dl_speed(dl, num_steps=10)
