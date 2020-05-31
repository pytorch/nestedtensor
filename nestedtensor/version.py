__version__ = '0.0.1.dev202053118+e3d48e4'
git_version = 'e3d48e4037c932ae4c91a44def2c4dd87fdd2955'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
