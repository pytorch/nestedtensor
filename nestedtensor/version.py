__version__ = '0.1.4+291a8a1'
git_version = '291a8a10d7de34c02ce2616db4eb8cf95ec27df9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
