__version__ = '0.0.1+7359688'
git_version = '73596883929d901edfdbb118316afe8e65d2b296'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
