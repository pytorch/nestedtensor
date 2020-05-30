__version__ = '0.0.1.dev20205301+ec19fc6'
git_version = 'ec19fc6a251b96ee661e60093dd8a1041f535d79'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
