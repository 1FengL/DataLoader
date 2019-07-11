import math
import os

import tensorflow as tf

from common import *
from parallel import MultiProcessDataset


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError("A Dataset must implement __getitem__(index) method.")

    def __len__(self):
        raise NotImplementedError("A Dataset must implement __len__() method.")

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __call__(self, *args, **kwargs):
        return self.__iter__()


class Dataloader(DatasetWrapper):
    def __init__(self,
                 ds,
                 output_types=None,
                 augmentations=None,
                 shuffle=False,
                 batch_size=1,
                 drop_remainder=True,
                 num_worker=os.cpu_count(),
                 num_prefetch=None,
                 transforms=None):

        super(Dataloader, self).__init__(ds)
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.num_worker = num_worker
        self.num_prefetch = num_worker if num_prefetch is None else num_prefetch
        self.transforms = transforms

        if self.augmentations is not None:
            self.ds = AugmentedDataset(self.ds, self.augmentations)

        if self.transforms is not None:
            self.ds = TransformedDataset(self.ds, self.transforms)
            # self.tfds = self.tfds.map(map_func=_Transforms(self.transforms), num_parallel_calls=num_map_worker)

        # TODO: auto adjust num_prefetch
        if self.num_worker > 1:
            self.ds = MultiProcessDataset(self.ds, num_worker=self.num_worker, num_prefetch=self.num_prefetch,
                                          shuffle=self.shuffle)
        elif self.shuffle:
            self.ds = ShuffledDataset(self.ds)

        if self.batch_size > 1:
            self.ds = BatchedDataset(self.ds, self.batch_size, drop_remainder=self.drop_remainder)

        # self.tfds = tf.data.Dataset.from_generator(self.ds, output_types=output_types)

        # if self.num_prefetch > 1:
        #     self.tfds = self.tfds.prefetch(num_prefetch)

    def __iter__(self):
        for dp in self.ds:
            yield dp

    # def __len__(self):
    #     try:
    #         if not self.drop_remainder:
    #             return math.ceil(len(self.ds) / self.batch_size)
    #         else:
    #             return len(self.ds) // self.batch_size
    #     except NotImplementedError:
    #         raise NotImplementedError("Dataset of Dataloader does not implement the __len__() method.")
