from dataloader.base import *


class SimpleDataset(Dataset):
    def __getitem__(self, index):
        if index > 9:
            raise IndexError("done")
        return index

    def __len__(self):
        return 10


def test_Dataloader():
    ds = SimpleDataset()
    dl = Dataloader(ds,
                    augmentations=None,
                    shuffle=False,
                    batch_size=2,
                    drop_remainder=True,
                    num_worker=2,
                    num_prefetch=None,
                    transforms=None)
    print("222")
    for epoch in range(2):
        for dp in dl:
            print(dp)
