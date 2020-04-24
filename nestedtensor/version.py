__version__ = '0.0.1.dev202042418+3ed649e'
git_version = '3ed649e9ead876d3ce403f1f85f9657bcc429e83'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
