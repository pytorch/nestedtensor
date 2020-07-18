__version__ = '0.0.1.dev20207184+c23b75d'
git_version = 'c23b75d7b967e2afed74e55ea5455707e94dc4db'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
