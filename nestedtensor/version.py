__version__ = '0.0.1.dev202022018+93fe178'
git_version = '93fe178d30a27fc1e5e8eace0d914a49baaee76f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
