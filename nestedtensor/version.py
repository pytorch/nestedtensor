__version__ = '0.0.1.dev20205161+876cd04'
git_version = '876cd04cb902c474fbc0900fdae3080c11a64df8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
