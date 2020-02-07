__version__ = '0.0.1.dev20202720+85889ad'
git_version = '85889ade876d7be34443bc8a562e811ba5cf2fe1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
