__version__ = '0.0.1.dev202012218+b60f22a'
git_version = 'b60f22afe9de975b8d6047623b8d29d95b385339'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
