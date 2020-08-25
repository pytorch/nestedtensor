__version__ = '0.0.1.dev20208252+7804c5c'
git_version = '7804c5c437b1e5f4736f4b01d624873d04940983'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
