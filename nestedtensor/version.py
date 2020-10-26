__version__ = '0.0.1.dev2020102622+c3f1e3e'
git_version = 'c3f1e3e28ac50e8608a311ad49440b9cdafeb3bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
