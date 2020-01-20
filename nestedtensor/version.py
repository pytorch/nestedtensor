__version__ = '0.0.1.dev20201207+05086b4'
git_version = '05086b497bc679a3daea0716d50d9e40d7cc40bc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
