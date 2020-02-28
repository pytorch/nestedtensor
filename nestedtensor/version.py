__version__ = '0.0.1.dev20202283+1ca6d3f'
git_version = '1ca6d3f7e9c243f160242867b0902eb7036d5bcf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
