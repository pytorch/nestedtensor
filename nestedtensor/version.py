__version__ = '0.0.1.dev202012322+b4e4097'
git_version = 'b4e40972179a55cbb1c4b32b1bf9b45f175a168a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
