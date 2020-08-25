__version__ = '0.0.1.dev20208253+eec9dfd'
git_version = 'eec9dfdf38998aa76b6182a60bdef4df6eb7796f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
