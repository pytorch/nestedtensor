__version__ = '0.0.1.dev20201175+d106c6f'
git_version = 'd106c6f533bbb15bc227e56d70af39ac63b59e1c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
