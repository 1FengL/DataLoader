from dataset.mnist import MNIST
from parallel import MultiProcessDataset


def test_mnist_shape():
    ds = MNIST(train_or_test='train')
    dl = MultiProcessDataset(ds, num_worker=2, num_prefetch=2)
    img_shape, label_shape = None, None
    for img, label in dl:
        img_shape = img.shape
        label_shape = label.shape
        break
    assert img_shape == (28, 28, 1)
    assert label_shape == ()

