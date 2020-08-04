__version__ = '0.0.1.dev20208420+0c8b1ee'
git_version = '0c8b1eec9ce778d556f7317b84c9bfaa85b32196'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
