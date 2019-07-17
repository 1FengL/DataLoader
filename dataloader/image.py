import cv2
import numpy as np

from dataloader.common import Transform


class Flip(Transform):
    def __init__(self, flipCode, data_index):
        """
        :param flipCode:
            1: horizontally
            0: vertically
            -1: horizontally and vertically
        """
        super(Flip, self).__init__(data_index)
        assert flipCode in [1, 0, -1]
        self.flipCode = flipCode

    def __call__(self, img):
        img_shape = img.shape
        result = cv2.flip(img, self.flipCode)
        if len(img_shape) == 3 and img_shape[2] == 1:
            result = np.expand_dims(result, 2)
        return result


def gray2rgb(img):
    return np.dstack([img] * 3)
