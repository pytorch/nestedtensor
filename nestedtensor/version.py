__version__ = '0.1.4+57dcab9'
git_version = '57dcab91aa657f6d8fa3f57c2ecf7a1556618655'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
