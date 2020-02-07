__version__ = '0.0.1.dev20202720+82f8e00'
git_version = '82f8e006534b69565ec7582956330d6e3236e137'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
