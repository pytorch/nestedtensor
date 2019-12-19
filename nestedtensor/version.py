__version__ = '0.0.1.dev201912193+9a5d8fa'
git_version = '9a5d8faf4a88f15e6870a8ceda16956714473cb3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
