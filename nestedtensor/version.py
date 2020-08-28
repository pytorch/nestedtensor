__version__ = '0.0.1.dev20208285+252e1ac'
git_version = '252e1ac8aceb9054dee7a5185221309781cce86e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
