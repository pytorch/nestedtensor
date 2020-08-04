__version__ = '0.0.1.dev20208321+632f5e6'
git_version = '632f5e6d9ce1c366f6a5e0dafe2696621d047d19'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
