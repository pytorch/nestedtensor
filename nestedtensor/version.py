__version__ = '0.0.1.dev202022722+a961a87'
git_version = 'a961a87e6b5343d770bc710ab7fad65a1ab5d209'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
