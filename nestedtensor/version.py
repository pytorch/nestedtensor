__version__ = '0.1.4+c9288b9'
git_version = 'c9288b93d3646e53ca4564794e3b8c1795798410'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
