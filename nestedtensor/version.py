__version__ = '0.0.1.dev202071422+00d020b'
git_version = '00d020b9bab8ac69433db78e6efc1aef44170b40'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
