__version__ = '0.0.1.dev202010233+b4190ef'
git_version = 'b4190efc91f3cd4891ae370502b656cbb63e7def'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
