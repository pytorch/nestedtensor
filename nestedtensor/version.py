__version__ = '0.0.1.dev202011123+8b33680'
git_version = '8b33680de269af825c3eec96a9d3841b4bef130f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
