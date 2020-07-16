__version__ = '0.0.1.dev202071620+95da3aa'
git_version = '95da3aae0059bb3cca091e8c9d170825a39ae0bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
