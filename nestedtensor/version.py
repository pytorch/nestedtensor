__version__ = '0.0.1.dev20208112+28af23f'
git_version = '28af23f62e5bd2837561405d348cced5d98f3f15'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
