__version__ = '0.0.1.dev201912173+d14cf1e'
git_version = 'd14cf1e8c83d12c6483992ced0cebbc66ffb6c41'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
