import math

import numpy as np
import tensorflow as tf

__all__ = ['DatasetWrapper', 'Transform', '_Transforms_for_tf_dataset',
           'BatchedDataset', 'TransformedDataset', 'ShuffledDataset',
           'AugmentedDataset']


class IndexableDatasetWrapper(object):
    def __init__(self, ds):
        self.ds = ds
        self.ds_len = len(ds)

    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def __len__(self):
        return len(self.ds)

    def __call__(self, *args, **kwargs):
        return self


class DatasetWrapper(object):
    def __init__(self, ds):
        self.ds = ds
        self.ds_len = len(ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for dp in self.ds:
            yield dp

    def __call__(self, *args, **kwargs):
        return self.__iter__()


class Transform(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Transform must implement __call__() method.")


class _Transforms_for_tf_dataset(object):
    """
    This class aggregate Transforms into one object in order to use tf.data.Dataset.map API
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        # assert len(args) == len(self.transforms)
        # data_list = [None] * len(args)
        # for i in range(len(args)):
        #     data = args[i]
        #     for transform in self.transforms[i]:
        #         data = transform(data)
        #     data_list[i] = data
        # return data_list
        data_list = list(args)
        for transform in self.transforms:
            data_list = transform(*data_list)
        return data_list


class BatchedDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 batch_size,
                 drop_remainder=True,
                 return_numpy=True,
                 keep_dims=True):
        super(BatchedDataset, self).__init__(ds)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.return_numpy = return_numpy
        self.keep_dims = keep_dims

    def __iter__(self):
        dp_buffer = []
        for dp in self.ds:
            dp_buffer.append(dp)
            if len(dp_buffer) == self.batch_size:
                yield self._batch_datapoints(dp_buffer)
                del dp_buffer[:]
        if not self.drop_remainder:
            yield self._batch_datapoints(dp_buffer)

    def __len__(self):
        ds_len = len(self.ds)
        if self.drop_remainder:
            return ds_len // self.batch_size
        else:
            return math.ceil(ds_len / self.batch_size)

    def _batch_datapoints(self, dp_buffer):
        """

        :param dp_buffer: a list of datapoints
        :return:
        """
        first_dp = dp_buffer[0]
        if isinstance(first_dp, (tuple, list)):
            dp_batch = [None] * len(first_dp)
            for i in range(len(first_dp)):
                dp_element_batch = []
                for j in range(len(dp_buffer)):
                    dp_element_batch.append(dp_buffer[j][i])
                if self.return_numpy:
                    dp_batch[i] = self._batch_ndarray(dp_element_batch)
                else:
                    dp_batch[i] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, dict):
            dp_batch = {}
            for key in first_dp.keys():
                dp_element_batch = []
                for j in range(len(dp_buffer)):
                    dp_element_batch.append(dp_buffer[j][key])
                if self.return_numpy:
                    dp_batch[key] = self._batch_ndarray(dp_element_batch)
                else:
                    dp_batch[key] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, np.ndarray):
            return self._batch_ndarray(dp_buffer)
        # single elements
        else:
            if self.return_numpy:
                return self._batch_ndarray(dp_buffer)
            else:
                return dp_buffer

    def _batch_ndarray(self, dp_element_batch):
        """

        :param dp_element_batch: a list of datapoint element, an element can be np.ndarray / list
        :return: np.ndarray, type is the same as input
        """
        try:
            ret = np.asarray(dp_element_batch)
            if self.keep_dims and len(ret.shape) == 1:
                ret = np.expand_dims(ret, 1)
            return ret
        except:
            raise ValueError("Unsupported type for batching.")


# class ShuffledDataset(DatasetWrapper):
#     def __init__(self, ds, buffer_size):
#         super(ShuffledDataset, self).__init__(ds)
#         self.buffer_size = buffer_size
#
#     def __iter__(self):
#         buffer = []
#         for dp in self.ds:
#             buffer.append(dp)
#             if len(buffer) == self.buffer_size:
#                 shuffled_idxs = np.random.permutation(self.buffer_size)
#                 for shuffled_idx in shuffled_idxs:
#                     yield buffer[shuffled_idx]
#                 del buffer[:]
#         if len(buffer) > 0:
#             shuffled_idxs = np.random.permutation(len(buffer))
#             for shuffled_idx in shuffled_idxs:
#                 yield buffer[shuffled_idx]
#             del buffer[:]


class ShuffledDataset(DatasetWrapper):
    def __init__(self, ds):
        super(ShuffledDataset, self).__init__(ds)

    def __iter__(self):
        self.shuffled_idxs = np.random.permutation(len(self.ds))
        for index, data in enumerate(self.ds):
            yield self.ds[self.shuffled_idxs[index]]


class TransformedDataset(IndexableDatasetWrapper):
    """

    """

    def __init__(self, ds, transforms):
        super(TransformedDataset, self).__init__(ds)
        self.transforms = transforms

    def __getitem__(self, index):
        dp = self.ds[index]
        for transform in self.transforms:
            assert callable(transform)
            if isinstance(dp, (list, tuple)):
                dp = transform(*dp)
            else:
                dp = transform(dp)
        return dp


class AugmentedDataset(IndexableDatasetWrapper):
    def __init__(self, ds, augmentations):
        super(AugmentedDataset, self).__init__(ds)
        self.augmentations = augmentations
        self.num_augmentations = len(self.augmentations)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError
        dp = self.ds[index % self.ds_len]
        if index < self.ds_len:
            return dp
        augmentation = self.augmentations[(index // self.ds_len) - 1]
        assert callable(augmentation)
        if isinstance(dp, (list, tuple)):
            return augmentation(*dp)
        else:
            return augmentation(dp)

    def __len__(self):
        # every augmentation gives one more duplication of dataset
        return self.ds_len * (1 + self.num_augmentations)
