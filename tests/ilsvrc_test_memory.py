import argparse
import psutil
import tensorflow as tf
import torch
from torch.utils.data import DataLoader as torchDataloader
from torch.utils.data import Dataset as torchDataset

from dataloader.common import Dataloader, TFDataloader
from dataloader.dataset import *
from dataloader.image import *
from dataloader.utils import *

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


def measure_dl_speed(dl_choice, num_steps, batch_size, num_worker, prepro, zmq, batchprefetch, warm_up=5):
    ds = ILSVRC12(path='/home/dsimsc/data/luoyifeng/ILSVRC12', train_or_test_or_val='train',
                  meta_dir='/home/dsimsc/data/luoyifeng/ILSVRC12', shape=(RESIZE_HEIGHT, RESIZE_WIDTH))
    if prepro:
        transform = [myTransform()]
    else:
        transform = None
    assert dl_choice in ('tf', 'tl', 'torch')
    if dl_choice == 'tf':
        dl = TFDataloader(ds, output_types=(tf.float32, tf.int32), shuffle=False, batch_size=batch_size,
                          transforms=transform)
    elif dl_choice == 'tl':
        dl = Dataloader(ds, output_types=(np.float32, np.int32), batch_size=batch_size, shuffle=False,
                        num_worker=num_worker, transforms=transform, use_zmq=zmq, prefetch_batch=batchprefetch)
    elif dl_choice == 'torch':
        if prepro:
            dl = torchDataloader(dl_for_torch(ds, myTransform()), batch_size=batch_size, shuffle=False,
                                 num_workers=num_worker)
        else:
            dl = torchDataloader(ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_worker)

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

    process = psutil.Process(os.getpid())
    start_memo = process.memory_info().rss
    print(start_memo)  # in bytes

    warm_up_cnt = 0
    loading_time_start = time.time()
    for img, label in dl:
        if warm_up_cnt < warm_up:
            pass
        else:
            loading_time_end = time.time()
            loading_time_sum += loading_time_end - loading_time_start

            rss_after, vms_after, shared_after = get_process_memory()
            rss_sum += rss_after - rss_before
            vms_sum += vms_after - vms_before
            shared_sum += shared_after - shared_before

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
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss - start_memo)  # in bytes
            cnt += 1
            if cnt == num_steps:
                break

        loading_time_start = time.time()
    print("Config -- dataloader choice: ", dl_choice, " | batch size: ", batch_size, " | num of worker: ", num_worker,
          " | preprocessing: ", prepro)
    print("Average loading: ", loading_time_sum / num_steps, " | Average training: ", training_time_sum / num_steps)
    print("Average RSS: ", format_bytes(rss_sum / num_steps), " | Average VMS: ",
          format_bytes(vms_sum / num_steps),
          " | Average SHR: ", format_bytes(shared_sum / num_steps))


class myTransform(Transform):

    def __call__(self, img, label):
        img = img[:, ::-1, :]
        img -= 127.5
        img /= 127.5
        return img, label


class dl_for_torch(torchDataset):
    def __init__(self, ds, prepro):
        self.ds = ds
        self.prepro = prepro

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        img, label = torch.from_numpy(img), torch.tensor(label)
        img = torch.flip(img, dims=[1])
        img -= 127.5
        img /= 127.5
        return img, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ILSVRC experiment arguments')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_worker', dest='num_worker', type=int, default=4)
    parser.add_argument('--num_steps', dest='num_steps', type=int, default=10)
    parser.add_argument('--dl', dest='dl_choice', type=str, choices=['tf', 'tl', 'torch'])
    parser.add_argument('--disable_prepro', dest='prepro', action='store_false')
    parser.add_argument('--disable_zmq', dest='zmq', action='store_false')
    parser.add_argument('--disable_batchprefetch', dest='batchprefetch', action='store_false')

    args = parser.parse_args()

    measure_dl_speed(dl_choice=args.dl_choice, num_steps=args.num_steps, num_worker=args.num_worker,
                     batch_size=args.batch_size, prepro=args.prepro, zmq=args.zmq, batchprefetch=args.batchprefetch)
