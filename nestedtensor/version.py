__version__ = '0.1.4+844b382'
git_version = '844b3824a777688155275cb6ad85fa51113e2f37'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
