from dataset.mnist import MNIST
from parallel import MultiProcessDataset


def test_mnist_shape():
    ds = MNIST(train_or_test='train', path='../data')
    img_shape, label_shape = None, None
    for img, label in ds:
        img_shape = img.shape
        label_shape = label.shape
        break
    assert img_shape == (28, 28, 1)
    assert label_shape == ()

