import sys

from base import *
from common import *


class SimpleDataset(Dataset):
    def __iter__(self):
        for i in range(100):
            yield i

    def __len__(self):
        return 100


def test_Dataloader():
    ds = SimpleDataset()
    dl = Dataloader(ds,
                    output_types=None,
                    augmentations=None,
                    shuffle=False,
                    shuffle_buffer_size=None,
                    batch_size=2,
                    drop_remainder=True,
                    num_extract_worker=2,
                    num_prefetch=None,
                    transforms=None)
    print()
    for epoch in range(2):
        for dp in dl:
            print(dp)
