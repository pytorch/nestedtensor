__version__ = '0.1.4+29dfd52'
git_version = '29dfd52677e1ff689af71c209dfa971dc8f142cb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
