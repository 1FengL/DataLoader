#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import logging

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['load_nietzsche_dataset']

NIETZSCHE_URL = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
NIETZSCHE_FILENAME = "nietzsche.txt"


def load_nietzsche_dataset(path='data', name='nietzsche'):
    """Load Nietzsche dataset.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/nietzsche/``.

    Returns
    --------
    str
        The content.

    Examples
    --------
    >>> see tutorial_generate_text.py
    >>> words = tl.files.load_nietzsche_dataset()
    >>> words = basic_clean_str(words)
    >>> words = words.split()

    """
    logging.info("Load or Download nietzsche dataset > {}".format(path))
    path = os.path.join(path, name)

    filepath = maybe_download_and_extract(NIETZSCHE_FILENAME, path, NIETZSCHE_URL)

    with open(filepath, "r") as f:
        words = f.read()
        return words


class NIETZSCHE(Dataset):

    def __init__(self, path='data', name='nietzsche'):
        self.words = load_nietzsche_dataset(path, name)

    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return len(self.words)
