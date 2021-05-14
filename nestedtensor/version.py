__version__ = '0.1.4+6e2417f'
git_version = '6e2417faa587b35cd5216d3c171311f6b0119d86'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
