__version__ = '0.0.1.dev202071018+ae638f4'
git_version = 'ae638f49dc6c5fb0394e217b4a43f54b17ec609f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
