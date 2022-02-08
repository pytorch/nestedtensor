__version__ = '0.1.4+77fcfc8'
git_version = '77fcfc851c5e742f1473b342e521d23ecca7109d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
