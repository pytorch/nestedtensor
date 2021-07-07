__version__ = '0.1.4+fc081fe'
git_version = 'fc081fe369c934261788a42e5d336f9469b884f9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
