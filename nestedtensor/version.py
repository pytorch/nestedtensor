__version__ = '0.0.1.dev20208185+841b659'
git_version = '841b6591df01ec5787b42b8b12a5604fb1d93305'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
