__version__ = '0.0.1.dev20206121+986ac07'
git_version = '986ac07b2267147cc0b0809d6d3cddf40cd2bacb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
