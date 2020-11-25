__version__ = '0.0.1.dev2020112522+61e4ea9'
git_version = '61e4ea92a32eb4e854175a82cce67a73f1945ffe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
