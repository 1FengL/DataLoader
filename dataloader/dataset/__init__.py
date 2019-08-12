from .mnist import MNIST
from .cifar10 import CIFAR10
from .ilsvrc import ILSVRC12, ILSVRC12Files, ILSVRCMeta

__all__ = ['MNIST', 'CIFAR10', 'ILSVRCMeta', 'ILSVRC12Files', 'ILSVRC12']

"""
Structure:
/path
    /name
        filename

Dataset / load_dataset: 
    - name: 
    - path: 

maybe_download_and_extract:
    - filename:
    - working_directory: = os.path.join(path, name)
    - url_source:

"""
