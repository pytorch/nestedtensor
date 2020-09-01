__version__ = '0.0.1.dev20209121+f761e62'
git_version = 'f761e62a46842775d8dc8519c4602e48c778df1c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
