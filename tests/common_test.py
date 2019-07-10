import math
import time
import numpy as np

from base import Dataset, Dataloader
from common import *
from dataset import MNIST


class SimpleDataset(Dataset):
    def __iter__(self):
        for i in range(5):
            yield i

    def __len__(self):
        return 5


class ExpensiveLoadingDataset(Dataset):
    def __iter__(self):
        for i in range(5):
            time.sleep(2)
            yield i

    def __len__(self):
        return 5


class plusx(Transform):
    def __init__(self, x=100):
        self.plus = x

    def __call__(self, x):
        return x + self.plus


def power2(x):
    return x ** 2


def add_label_value_to_img(img, label):
    return img + label, label


def test_BatchedDataset():
    ds = SimpleDataset()
    dl = BatchedDataset(ds, batch_size=2, drop_remainder=True)
    assert len(dl) == 2
    for i, batch in enumerate(dl):
        assert len(batch) == 2
        if i == 0:
            assert batch[0] == 0 and batch[1] == 1
        if i == 1:
            assert batch[0] == 2 and batch[1] == 3
    assert i == 1
    dl = BatchedDataset(ds, batch_size=2, drop_remainder=False)
    assert len(dl) == 3
    for i, batch in enumerate(dl):
        if i == 0:
            assert batch[0] == 0 and batch[1] == 1
        if i == 1:
            assert batch[0] == 2 and batch[1] == 3
        if i == 2:
            assert batch[0] == 4
    assert i == 2

    ds = MNIST(train_or_test='train', path='../data')
    dl = BatchedDataset(ds, batch_size=500, drop_remainder=True)
    assert len(dl) == 120  # 60000 / 500 = 120
    img_shape, label_shape = None, None
    for img, label in dl:
        img_shape = img.shape
        label_shape = label.shape
        break
    assert img_shape == (500, 28, 28, 1)
    assert label_shape == (500,)


def test_ShuffledDataset():
    ds = SimpleDataset()
    dl = ShuffledDataset(ds, buffer_size=10)
    dps = []
    for dp in dl:
        dps.append(dp)
    assert set(dps) == set([0, 1, 2, 3, 4])
    print("\n######### test_ShuffledDataset() #########")
    print("shuffled dataset is: ", dps)


def test_TransformedDataset():
    ds = MNIST(train_or_test='train', path='../data')
    dl = TransformedDataset(ds, transforms=[(plusx(x=999), 0)])
    for img, label in dl:
        # pixel value is between [0, 1], should be [999, 1000] when added 999
        assert img[0, 0, 0] >= 999
        assert img[0, 0, 0] <= 1000

    dl = TransformedDataset(ds, transforms=[[plusx(x=999), 0], (power2, 1)])
    for img, label in dl:
        assert label == int(math.sqrt(label)) ** 2

    dl = TransformedDataset(ds, transforms=[add_label_value_to_img])
    for img, label in dl:
        # pixel value is between [0, 1]
        assert img[0, 0, 0] - label >= 0
        assert img[0, 0, 0] - label <= 1


def test_AugmentedDataset():
    ds = MNIST(train_or_test='train', path='../data')
    dl = AugmentedDataset(ds, augmentations=[(plusx(x=999), 1)])
    assert len(dl) == 120000  # 60000 * 2 = 120000
    cnt = 0
    for img, label in dl:
        if label >= 999:
            cnt += 1
    assert cnt == 60000
