__version__ = '0.0.1.dev20202421+77c61a3'
git_version = '77c61a316bc64ca416366e4e5e59da814a42ff61'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
