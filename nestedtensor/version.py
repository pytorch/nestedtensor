from __future__ import absolute_import, division, print_function, unicode_literals


__version__ = '0.0.1.dev2020254+f5337d6'
git_version = 'f5337d6604e2c8e9ce9fb2795c9f335c4fee4a22'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
