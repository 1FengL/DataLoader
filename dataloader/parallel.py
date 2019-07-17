import multiprocessing
import numpy as np

from dataloader.common import DatasetWrapper


class MultiProcessDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 num_worker,
                 num_prefetch,
                 shuffle=False):

        super(MultiProcessDataset, self).__init__(ds)
        self.num_worker = num_worker
        self.num_prefetch = num_prefetch
        self.shuffle = shuffle

        self.index_queue = multiprocessing.Queue(self.num_worker)
        self.data_queue = multiprocessing.Queue(self.num_prefetch)
        self.put_idx_worker = None
        for _ in range(num_worker):
            worker = multiprocessing.Process(target=self._worker,
                                             args=(self.ds, self.index_queue, self.data_queue))
            worker.daemon = True
            worker.start()

    def _worker(self, ds, index_q, data_q):
        while True:
            idx = index_q.get()
            data_q.put((idx, ds[idx]))

    def _put_idxs(self, idxs, index_q):
        for idx in idxs:
            index_q.put(idx)

    def __iter__(self):
        # shutdown put_idx_worker and clear queues from previous epoch
        _shutdown_proc(self.put_idx_worker)
        while not self.index_queue.empty():
            self.index_queue.get()
        while not self.data_queue.empty():
            self.data_queue.get()

        # shuffle at the start of every epoch
        if self.shuffle:
            self.idxs = np.random.permutation(self.ds_len)
        else:
            self.idxs = np.arange(self.ds_len)

        self.put_idx_worker = multiprocessing.Process(target=self._put_idxs,
                                                      args=(self.idxs, self.index_queue))
        self.put_idx_worker.daemon = True
        self.put_idx_worker.start()

        data_buffer = {}
        for return_idx in self.idxs:
            if return_idx in data_buffer:
                yield data_buffer.pop(return_idx)
            else:
                while True:
                    idx, dp = self.data_queue.get()
                    if idx == return_idx:
                        yield dp
                        break
                    else:
                        data_buffer[idx] = dp
        _shutdown_proc(self.put_idx_worker)


def _shutdown_proc(proc):
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join()

# class MultiProcessDataset(DatasetWrapper):
#     def __init__(self,
#                  ds,
#                  num_worker,
#                  num_prefetch):
#         super(MultiProcessDataset, self).__init__(ds)
#         self.num_worker = num_worker
#         self.num_prefetch = num_prefetch
#
#         try:
#             self.ds_len = len(self.ds)
#         except NotImplementedError:
#             self.ds_len = -1
#
#         self.data_queue = multiprocessing.Queue(self.num_prefetch)
#         for _ in range(num_worker):
#             worker = multiprocessing.Process(target=self._worker,
#                                              args=(self.ds, self.data_queue))
#             worker.start()
#             atexit.register(stop_proc_by_weak_ref, weakref.ref(worker))
#
#     def _worker(self, ds, q):
#         while True:
#             for dp in ds:
#                 q.put(dp)
#
#     def __iter__(self):
#         cnt = 0
#         while True:
#             cnt += 1
#             yield self.data_queue.get()
#             if (self.ds_len > 0) and (cnt >= self.ds_len):
#                 break
#
#
# def stop_proc_by_weak_ref(ref):
#     proc = ref()
#     if proc is None:
#         return
#     if not proc.is_alive():
#         return
#     proc.terminate()
#     proc.join()
