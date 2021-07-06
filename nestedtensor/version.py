__version__ = '0.1.4+50379e2'
git_version = '50379e231a3f2b5e97f51353cd2649cb86bc8290'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
