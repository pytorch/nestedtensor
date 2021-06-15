__version__ = '0.1.4+431672a'
git_version = '431672a15f8be60f786afd0d73bd175ba3bf44fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
