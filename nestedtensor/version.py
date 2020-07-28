__version__ = '0.0.1.dev202072819+a9a3ea9'
git_version = 'a9a3ea94aaa884a6e4de085dedd8d61a16988df4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
