__version__ = '0.0.1.dev202081723+c14f3e2'
git_version = 'c14f3e232d769b37b0942ba330d7975f4b133ebd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
