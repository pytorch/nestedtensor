__version__ = '0.1.4+44e5453'
git_version = '44e545372f4516c544c56046b196696ad58f7098'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
