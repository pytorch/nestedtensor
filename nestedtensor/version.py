__version__ = '0.0.1.dev202082421+93f26bc'
git_version = '93f26bcc6cd53fbf7644909e148efcdf2f6f49a3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
