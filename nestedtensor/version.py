__version__ = '0.0.1.dev202011112+7418a43'
git_version = '7418a43aa0aebff7c4e2fb31655772f61842e4dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
