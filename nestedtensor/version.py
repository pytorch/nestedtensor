__version__ = '0.0.1.dev202051617+fd32728'
git_version = 'fd3272839da9b0dcd5184190cf332287539dc670'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
