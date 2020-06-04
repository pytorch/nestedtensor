__version__ = '0.0.1.dev20206420+bdadcbf'
git_version = 'bdadcbf671dc3c4490ed2adbd773c82e72b9cc94'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
