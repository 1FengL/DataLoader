import math

from dataloader.base import *
from dataloader.common import *
from dataloader.dataset import MNIST


class SimpleDataset(Dataset):
    def __getitem__(self, index):
        if index > 4:
            raise IndexError("done")
        return index

    def __len__(self):
        return 5


class plusx(Transform):
    def __init__(self, x=100):
        self.plus = x

    def __call__(self, x):
        return x + self.plus


class mnist_plusx(Transform):
    def __init__(self, x=100):
        self.plus = x

    def __call__(self, img, label):
        return img, label + self.plus


class mnist_label_power2(Transform):
    def __call__(self, img, label):
        return img, label ** 2


def power2(x):
    return x ** 2


def mnist_preprocessing(img, label):
    return img + label, label


def test_BatchedDataset():
    ds = SimpleDataset()
    dl = PrefetchBatchedDataset(ds, batch_size=2, drop_remainder=True)
    assert len(dl) == 2
    for i, batch in enumerate(dl):
        assert len(batch) == 2
        if i == 0:
            assert batch[0] == 0 and batch[1] == 1
        if i == 1:
            assert batch[0] == 2 and batch[1] == 3
    assert i == 1
    dl = PrefetchBatchedDataset(ds, batch_size=2, drop_remainder=False)
    assert len(dl) == 3
    for i, batch in enumerate(dl):
        if i == 0:
            assert batch[0] == 0 and batch[1] == 1
        if i == 1:
            assert batch[0] == 2 and batch[1] == 3
        if i == 2:
            assert batch[0] == 4
    assert i == 2

    ds = MNIST(train_or_test='train', path='../data', shape=(-1, 28, 28, 1))
    dl = PrefetchBatchedDataset(ds, batch_size=500, drop_remainder=True)
    assert len(dl) == 120  # 60000 / 500 = 120
    img_shape, label_shape = None, None
    for img, label in dl:
        img_shape = img.shape
        label_shape = label.shape
        break
    assert img_shape == (500, 28, 28, 1)
    assert label_shape == (500,)


# def test_ShuffledDataset():
#     ds = SimpleDataset()
#     dl = ShuffledDataset(ds, buffer_size=10)
#     dps = []
#     for dp in dl:
#         dps.append(dp)
#     assert set(dps) == set([0, 1, 2, 3, 4])
#     print("\n######### test_ShuffledDataset() #########")
#     print("shuffled dataset is: ", dps)


def test_TransformedDataset():
    ds = SimpleDataset()
    dl = TransformedDataset(ds, transforms=[plusx(x=999)])
    result = [item for item in dl]
    assert set(result) == {999, 1000, 1001, 1002, 1003}

    ds = MNIST(train_or_test='train', path='../data', shape=(-1, 28, 28, 3))
    dl = TransformedDataset(ds, transforms=[mnist_label_power2()])
    for img, label in dl:
        assert label == int(math.sqrt(label)) ** 2

    dl = TransformedDataset(ds, transforms=[mnist_preprocessing])
    for img, label in dl:
        # pixel value is between [0, 1]
        assert img[0, 0, 0] - label >= 0
        assert img[0, 0, 0] - label <= 1


def test_AugmentedDataset():
    ds = MNIST(train_or_test='train', path='../data')
    dl = AugmentedDataset(ds, augmentations=[mnist_plusx(x=999)])
    assert len(dl) == 120000  # 60000 * 2 = 120000
    cnt = 0
    for img, label in dl:
        if label >= 999:
            cnt += 1
    assert cnt == 60000
