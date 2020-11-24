__version__ = '0.0.1.dev2020112422+f1bcae2'
git_version = 'f1bcae20c8ba8c0de9d48e71b221221f2893f7eb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
