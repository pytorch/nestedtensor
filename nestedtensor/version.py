__version__ = '0.0.1.dev20208293+6ba4f2f'
git_version = '6ba4f2f76fc8915363a7f6c26a5411562d143f89'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
