__version__ = '0.0.1.dev20208260+33227d9'
git_version = '33227d99aacb2f2d03896f0af1d7daa350270df4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
