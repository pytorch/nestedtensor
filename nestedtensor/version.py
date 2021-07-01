__version__ = '0.1.4+77d3e21'
git_version = '77d3e21c586971b84b5281cae031b9ec23673673'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
