__version__ = '0.1.4+9b8cb54'
git_version = '9b8cb548035758286cc03c1a69ef622b6b5f09aa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
