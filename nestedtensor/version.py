__version__ = '0.0.1.dev202091619+a397be5'
git_version = 'a397be58cd805c28e0563ffc8e0ec9115896959a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
