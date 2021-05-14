__version__ = '0.1.4+a70e400'
git_version = 'a70e4005507d9cb7fdc213417f569901804c56f0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
