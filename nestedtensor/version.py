__version__ = '0.0.1.dev202011320+09df07f'
git_version = '09df07f0347f25e260b166b8e3a1623d5980411d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
