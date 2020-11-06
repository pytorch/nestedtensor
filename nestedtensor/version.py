__version__ = '0.0.1.dev20201166+e92b81f'
git_version = 'e92b81f6abfd6260222338eb1f4a49b35a7562ac'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
