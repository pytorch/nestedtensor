__version__ = '0.1.4+82420f2'
git_version = '82420f2689ee9d60fb5023f8fea97ecd1f93bd29'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
