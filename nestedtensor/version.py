__version__ = '0.1.4+13fdbc8'
git_version = '13fdbc8624f32a660b2ada11b0b2be8fa50ab9aa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
