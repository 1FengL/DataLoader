import atexit
import multiprocessing
import weakref

from common import DatasetWrapper


class MultiProcessDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 num_worker,
                 num_prefetch):
        super(MultiProcessDataset, self).__init__(ds)
        self.num_worker = num_worker
        self.num_prefetch = num_prefetch

        try:
            self.ds_len = len(self.ds)
        except NotImplementedError:
            self.ds_len = -1

        self.data_queue = multiprocessing.Queue(self.num_prefetch)
        for _ in range(num_worker):
            worker = multiprocessing.Process(target=self._worker,
                                             args=(self.ds, self.data_queue))
            worker.start()
            atexit.register(stop_proc_by_weak_ref, weakref.ref(worker))

    def _worker(self, ds, q):
        while True:
            for dp in ds:
                q.put(dp)

    def __iter__(self):
        cnt = 0
        while True:
            cnt += 1
            yield self.data_queue.get()
            if (self.ds_len > 0) and (cnt >= self.ds_len):
                break


def stop_proc_by_weak_ref(ref):
    proc = ref()
    if proc is None:
        return
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join()
