import gzip
import logging
import os
import numpy as np

from base import Dataset
from utils import maybe_download_and_extract


class MNIST(Dataset):
    def __init__(self, train_or_test, path='data', name='mnist', url='http://yann.lecun.com/exdb/mnist/'):
        self.url = url
        self.path = os.path.join(path, name)

        assert train_or_test in ['train', 'test']
        # Download and read the training and test set images and labels.
        logging.info("Load or Download {0} > {1}".format(name.upper(), self.path))
        if train_or_test == 'train':
            self.images = self.load_mnist_images('train-images-idx3-ubyte.gz')
            self.labels = self.load_mnist_labels('train-labels-idx1-ubyte.gz')
        else:
            self.images = self.load_mnist_images('t10k-images-idx3-ubyte.gz')
            self.labels = self.load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    def load_mnist_images(self, filename):
        filepath = maybe_download_and_extract(filename, self.path, self.url)

        logging.info(filepath)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape((-1, 28, 28, 1))
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(self, filename):
        filepath = maybe_download_and_extract(filename, self.path, self.url)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    def __len__(self):
        return self.images.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self.images[i], self.labels[i]
