__version__ = '0.0.1.dev20208616+a839cf7'
git_version = 'a839cf72a5ad36e81e9b3ac4f9004b044aa5419a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
