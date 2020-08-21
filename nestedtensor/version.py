__version__ = '0.0.1.dev20208214+fe1a331'
git_version = 'fe1a331a5f4c0143881be331fd5bf27ef8920d3a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
