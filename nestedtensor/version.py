__version__ = '0.0.1.dev20206123+8afa4aa'
git_version = '8afa4aa4c01289b77113ba53b8f36f6f65edab98'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
