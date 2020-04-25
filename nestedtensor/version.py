__version__ = '0.0.1.dev20204254+df4fc3a'
git_version = 'df4fc3a4557b0bbbe018ff67659794f469bee585'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
