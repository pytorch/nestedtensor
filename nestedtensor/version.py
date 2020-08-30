__version__ = '0.0.1.dev20208305+9b59c16'
git_version = '9b59c1662f99295754481939094f913183130209'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
