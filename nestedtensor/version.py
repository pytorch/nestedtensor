__version__ = '0.1.4+169ef86'
git_version = '169ef86b665240266a21a4d9a5407bc033f688c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
