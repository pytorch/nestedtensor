__version__ = '0.0.1.dev20201163+f6f01cd'
git_version = 'f6f01cd367b1b321aa46fe956b2c98c3b8d24cd3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
