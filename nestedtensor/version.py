__version__ = '0.1.4+81f953a'
git_version = '81f953aba3902318eb8af7a9e6e1cc8fe254cda8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
