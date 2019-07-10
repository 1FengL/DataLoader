import math

import numpy as np

__all__ = ['DatasetWrapper', 'Transform', '_Transforms_for_tf_dataset',
           'BatchedDataset', 'ShuffledDataset', 'TransformedDataset',
           'AugmentedDataset']


class DatasetWrapper(object):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for data in self.ds:
            yield data

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
            data_list[transform.data_index] = transform(data_list[transform.data_index])
        return data_list


class BatchedDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 batch_size,
                 drop_remainder=True,
                 return_numpy=True):
        super(BatchedDataset, self).__init__(ds)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.return_numpy = return_numpy

    def __iter__(self):
        dp_buffer = []
        for dp in self.ds:
            dp_buffer.append(dp)
            if len(dp_buffer) == self.batch_size:
                yield BatchedDataset._batch_datapoints(dp_buffer, self.return_numpy)
                del dp_buffer[:]
        if not self.drop_remainder:
            yield BatchedDataset._batch_datapoints(dp_buffer, self.return_numpy)

    def __len__(self):
        ds_len = len(self.ds)
        if self.drop_remainder:
            return ds_len // self.batch_size
        else:
            return math.ceil(ds_len / self.batch_size)

    @staticmethod
    def _batch_datapoints(dp_buffer, return_numpy=True):
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
                if return_numpy:
                    dp_batch[i] = BatchedDataset._batch_ndarray(dp_element_batch)
                else:
                    dp_batch[i] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, dict):
            dp_batch = {}
            for key in first_dp.keys():
                dp_element_batch = []
                for j in range(len(dp_buffer)):
                    dp_element_batch.append(dp_buffer[j][key])
                if return_numpy:
                    dp_batch[key] = BatchedDataset._batch_ndarray(dp_element_batch)
                else:
                    dp_batch[key] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, np.ndarray):
            return BatchedDataset._batch_ndarray(dp_buffer)
        # single elements
        else:
            if return_numpy:
                return BatchedDataset._batch_ndarray(dp_buffer)
            else:
                return dp_buffer

    @staticmethod
    def _batch_ndarray(dp_element_batch):
        """

        :param dp_element_batch: a list of datapoint element, an element can be np.ndarray / list
        :return: np.ndarray, type is the same as input
        """
        try:
            return np.asarray(dp_element_batch)
        except:
            raise ValueError("Unsupported type for batching.")


class ShuffledDataset(DatasetWrapper):
    def __init__(self, ds, buffer_size):
        super(ShuffledDataset, self).__init__(ds)
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for dp in self.ds:
            buffer.append(dp)
            if len(buffer) == self.buffer_size:
                shuffled_idxs = np.random.permutation(self.buffer_size)
                for shuffled_idx in shuffled_idxs:
                    yield buffer[shuffled_idx]
                del buffer[:]
        if len(buffer) > 0:
            shuffled_idxs = np.random.permutation(len(buffer))
            for shuffled_idx in shuffled_idxs:
                yield buffer[shuffled_idx]
            del buffer[:]


class TransformedDataset(DatasetWrapper):
    """

    """

    def __init__(self, ds, transforms):
        super(TransformedDataset, self).__init__(ds)
        self.transforms = transforms

    def __iter__(self):
        for dp in self.ds:
            dp = list(dp)
            for transform in self.transforms:
                if isinstance(transform, (tuple, list)):
                    assert callable(transform[0])
                    dp[transform[1]] = transform[0](dp[transform[1]])
                else:
                    assert callable(transform)
                    dp = transform(*dp)
            yield tuple(dp)


class AugmentedDataset(DatasetWrapper):
    def __init__(self, ds, augmentations):
        super(AugmentedDataset, self).__init__(ds)
        self.augmentations = augmentations

    def __len__(self):
        # every augmentation gives one more duplication of dataset
        return len(self.ds) * (1 + len(self.augmentations))

    def __iter__(self):
        for dp in self.ds:
            yield dp
            dp = list(dp)
            for augmentation in self.augmentations:
                if isinstance(augmentation, (tuple, list)):
                    assert callable(augmentation[0])
                    dp[augmentation[1]] = augmentation[0](dp[augmentation[1]])
                else:
                    assert callable(augmentation)
                    dp = augmentation(*dp)
            yield tuple(dp)
