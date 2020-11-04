__version__ = '0.0.1.dev202011321+05ddc00'
git_version = '05ddc0095dc99259c21cab4472002758a3f63502'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
