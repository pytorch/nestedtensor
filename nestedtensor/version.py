__version__ = '0.0.1.dev202011918+2bc08df'
git_version = '2bc08df4b377122f6c8a826dfd508229958837c0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
