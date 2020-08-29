__version__ = '0.0.1.dev20208294+7666ef0'
git_version = '7666ef031d5492e74f64a9cc2be297db8e6e733f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
