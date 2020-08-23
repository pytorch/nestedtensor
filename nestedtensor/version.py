__version__ = '0.0.1.dev202082320+227ede3'
git_version = '227ede3355e228b8def91b39036df6f8c602e8d1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
